import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import json

import joblib
from joblib import Parallel, delayed

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, normalize

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import math

from numerapi import NumerAPI



print("Modulos cargardos")


# Function to pad sequences efficiently
def pad_sequence(inputs, padding_value=-1, max_len=None):
    if max_len is None:
        max_len = max([input.shape[0] for input in inputs])
    padded_inputs = torch.full((len(inputs), max_len, inputs[0].shape[1]), padding_value, dtype=inputs[0].dtype)
    masks = torch.zeros((len(inputs), max_len, 1), dtype=torch.float)

    for i, input in enumerate(inputs):
        pad_len = input.shape[0]
        padded_inputs[i, :pad_len] = input
        masks[i, :pad_len] = 1
    
    return padded_inputs, masks

# Convert data to torch and pad sequences
def convert_to_torch(era, data, feature_names, target_names, padding_value, max_len):
    inputs = torch.from_numpy(data[feature_names].values.astype(np.int8))
    labels = torch.from_numpy(data[target_names].values.astype(np.float32))

    padded_inputs, masks_inputs = pad_sequence([inputs], padding_value=padding_value, max_len=max_len)
    padded_labels, masks_labels = pad_sequence([labels], padding_value=padding_value, max_len=max_len)

    return {
        era: (padded_inputs, padded_labels, masks_inputs)
    }

# Main function to get era to data mapping
def get_era2data(df, feature_names, target_names, padding_value=-1, max_len=None):
    # Parallel processing to convert and pad data for each era
    res = Parallel(n_jobs=-1, prefer="threads")(
        delayed(convert_to_torch)(era, data, feature_names, target_names, padding_value, max_len)
        for era, data in tqdm(df.groupby("era_int"))
    )
    # Merge results into a single dictionary
    era2data = {era: data for r in res for era, data in r.items()}
    return era2data


# Clase para la codificación posicional, añadiendo información de posición a los embeddings de entrada
class PositionalEncoding(nn.Module):
    """
    Agrega codificación posicional a la entrada para incorporar información de la posición.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Crear un tensor para la codificación posicional
        pe = torch.zeros(max_len, d_model)
        # Calcular las posiciones y los términos de división basados en la fórmula estándar de codificación posicional
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        # Asignar funciones seno y coseno a los elementos de posición par e impar
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Registrar el buffer para usarlo durante el entrenamiento sin ser un parámetro entrenable
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Sumar la codificación posicional a la entrada y aplicar dropout
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

# Capa de red feedforward que aplica una transformación lineal y activación no lineal
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.1):
        super().__init__()
        # Definir las capas lineales y dropout
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Aplicar transformaciones con ReLU y Dropout
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# Clase de atención modular, que soporta diferentes tipos de atención
class ModularAttention(nn.Module):
    def __init__(self, d_model, attention_type="linear", dropout=0.1, window_size=1000):
        super().__init__()
        self.dim = d_model
        self.attention_type = attention_type
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, k, q, v, mask=None):
        # Comprobar que las dimensiones de k, q y v coincidan
        assert k.size(0) == q.size(0)
        assert k.size(0) == v.size(0)
        n = math.sqrt(self.dim)

        if self.attention_type == "vanilla":
            # Aplicar la atención clásica
            scores = self.sliding_window_scores(q, k, n)  # Usa la función de ventana deslizante
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        elif self.attention_type == "linear":
            # Aplicar la atención lineal
            scores = torch.matmul(k.transpose(-2, -1), v) / n
            if mask is not None:
                mask = mask.unsqueeze(-1)
                q = q.masked_fill(mask == 0, float("-inf"))
            denom = 1 + torch.sum(torch.exp(q), dim=1, keepdim=True)
            q = torch.exp(q) / denom
            q = self.dropout(F.softmax(q, dim=1))
            out = torch.matmul(q, scores)

        return out

    # Función para calcular scores con ventana deslizante
    def sliding_window_scores(self, q, k, scale):
        batch_size, num_heads, num_queries, head_dim = q.size()
        _, _, num_keys, _ = k.size()

        # Inicializar los scores como ceros
        scores = torch.zeros(batch_size, num_heads, num_queries, num_keys, device=q.device)
        for i in range(0, num_queries, self.window_size):
            # Definir los límites de la ventana
            start = max(0, i - self.window_size // 2)
            end = min(num_keys, i + self.window_size // 2 + 1)

            # Calcular los scores dentro de la ventana
            q_slice = q[:, :, i:i+1, :]
            k_slice = k[:, :, start:end, :].transpose(2, 3)
            window_scores = torch.matmul(q_slice, k_slice) / scale
            scores[:, :, i, start:end] = window_scores.squeeze(2)

        return scores

# Clase para la atención multi-cabezal con atención modular
class MultiHeadModularAttention(nn.Module):
    def __init__(self, d_model, num_heads, attention_type="linear", dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ModularAttention(d_model, attention_type, dropout)
        self.head_dim = d_model // num_heads

        # Transformaciones lineales para las llaves, consultas y valores
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Transformación lineal para la salida
        self.output_linear = nn.Linear(d_model, d_model)

        # Dropout y normalización de capas
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    # Función para dividir las entradas en múltiples cabezas
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim).clone()
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(0)
        k = self.linear_k(inputs)
        q = self.linear_q(inputs)
        v = self.linear_v(inputs)

        # Dividir llaves, consultas y valores en múltiples cabezas
        k, q, v = [self.split_heads(x, batch_size) for x in [k, q, v]]

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Aplicar el mecanismo de atención
        attn_output = self.attention(k, q, v, mask)
        # Reestructurar el tensor de salida
        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .clone()
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        # Aplicar transformación lineal y normalización de capas
        output = self.output_linear(attn_output)
        output = self.layer_norm(output)

        return output

# Clase para la codificación del transformador
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_heads,
        num_layers,
        dropout_prob=0.15,
        max_len=5000,
    ):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.d_model = d_model

        # Codificación posicional y capa inicial de mapeo
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
        )
        # Capas del codificador
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    MultiHeadModularAttention(
                        d_model, num_heads, attention_type="vanilla"
                    ),
                    nn.LayerNorm(d_model),
                    FeedForwardLayer(d_model=d_model),
                    nn.Dropout(dropout_prob),
                )
                for _ in range(num_layers)
            ]
        )
        # Capa para mapear la entrada al modelo de dimensión deseada
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.Linear(d_model, d_model)
        )

    def forward(self, inputs, mask=None):
        x = self.mapper(inputs)

        # Agregar la codificación posicional
        x = x + self.positional_encoding(x)

        # Aplicar las capas del codificador
        for layer in self.layers:
            layer_output = layer[0](x, mask)

            for sublayer in layer[1:]:
                layer_output = sublayer(layer_output)

            x = x + layer_output

        return x

# Clase principal del Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_heads,
        num_layers,
        dropout_prob=0.15,
        max_len=6000,
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.d_model = d_model

        # Codificador del transformador
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
        )
        # Capas finales de la red neuronal
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SELU(),
            nn.Linear(d_model // 2, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, inputs, mask=None):
        emb = self.encoder(inputs, mask)
        outputs = self.fc(emb)
        return outputs
    

def pearsonr(x, y, eps=1e-8):

    # Calcular las medias de x e y
    mx = torch.mean(x)
    my = torch.mean(y)

    # Restar las medias
    xm = x - mx
    ym = y - my

    # Calcular la suma de productos y las sumas de cuadrados
    r_num = torch.dot(xm, ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + eps

    # Calcular la correlación de Pearson
    r = r_num / r_den

    # Devolver el coeficiente de correlación de Pearson
    return r



def calculate_loss(outputs, padded_labels, masks_inputs, target_weight_softmax=None):
    """
    Calcula la pérdida usando MSE y correlación de Pearson entre los outputs y las etiquetas.

    Args:
        outputs (torch.Tensor): Salidas del modelo.
        padded_labels (torch.Tensor): Etiquetas reales ajustadas.
        masks_inputs (torch.Tensor): Máscaras que indican los valores válidos.
        target_weight_softmax (torch.Tensor, opcional): Ponderaciones para los objetivos.

    Returns:
        tuple: pérdida total, MSE, y correlación de Pearson.
    """
    # Expandir dimensiones para coincidir con los outputs
    masks_inputs = masks_inputs.unsqueeze(-1)

    # Calcular el error cuadrático medio (MSE) ponderado si aplica
    if target_weight_softmax is not None:
        _mse = criterion(
            outputs * masks_inputs * target_weight_softmax,
            padded_labels * masks_inputs * target_weight_softmax
        )
    else:
        _mse = criterion(outputs * masks_inputs, padded_labels * masks_inputs)

    # Adicionar el MSE de la primera etiqueta (principal) y ajustar ponderación
    _mse = _mse * 0.1 + criterion(outputs[:, 0] * masks_inputs, padded_labels[:, 0] * masks_inputs)

    # Calcular correlación de Pearson solo con el objetivo principal
    valid_indices = masks_inputs.view(-1).nonzero().squeeze()
    corr = pearsonr(
        outputs[0][:, 0][valid_indices],
        padded_labels[0][:, 0][valid_indices],
    )

    # Pérdida total ajustada por correlación
    loss = _mse - corr
    return loss, _mse, corr

def process_batch(batch, device):
    """
    Prepara y transfiere los datos del batch al dispositivo.

    Args:
        batch (tuple): Batch de datos (inputs, labels, masks).
        device (str): Dispositivo de cómputo (CPU o GPU).

    Returns:
        tuple: Inputs, labels, y masks procesados y enviados al dispositivo.
    """
    padded_inputs = batch[0].to(device=device)
    padded_labels = batch[1].to(device=device)
    masks_inputs = batch[2].to(device=device).squeeze(-1)
    return padded_inputs, padded_labels, masks_inputs

def train_on_batch(transformer, optimizer, batch, device):
    """
    Realiza el entrenamiento en un batch de datos.

    Args:
        transformer (nn.Module): Modelo de Transformer.
        optimizer (torch.optim.Optimizer): Optimizador.
        batch (tuple): Batch de datos (inputs, labels, masks).
        device (str): Dispositivo de cómputo.

    Returns:
        tuple: Pérdida, MSE, y correlación de Pearson.
    """
    # Preparar los datos del batch
    padded_inputs, padded_labels, masks_inputs = process_batch(batch, device)

    optimizer.zero_grad()

    # Realizar la normalización de inputs y obtener salidas del modelo
    outputs = transformer(padded_inputs / 4.0, masks_inputs)

    # Generar pesos aleatorios y normalizarlos con softmax
    random_weights = torch.rand(padded_labels.shape[-1], device=device)
    target_weight_softmax = F.softmax(random_weights, dim=0)

    # Calcular la pérdida y realizar el paso de backpropagation
    loss, _mse, _corr = calculate_loss(outputs, padded_labels, masks_inputs, target_weight_softmax)
    loss.backward()
    optimizer.step()
    return loss.item(), _mse.item(), _corr.item()

def evaluate_on_batch(transformer, batch, device):
    """
    Evalúa el modelo en un batch de datos.

    Args:
        transformer (nn.Module): Modelo de Transformer.
        batch (tuple): Batch de datos (inputs, labels, masks).
        device (str): Dispositivo de cómputo.

    Returns:
        tuple: Pérdida, MSE, y correlación de Pearson.
    """
    padded_inputs, padded_labels, masks_inputs = process_batch(batch, device)
    outputs = transformer(padded_inputs / 4.0, masks_inputs)
    loss, _mse, _corr = calculate_loss(outputs, padded_labels, masks_inputs)
    return loss.item(), _mse.item(), _corr.item()

def train_model(transformer, optimizer, num_epochs, train_loader, val_loader, device='cuda'):
    """
    Entrena y evalúa el modelo en múltiples épocas.

    Args:
        transformer (nn.Module): Modelo de Transformer.
        optimizer (torch.optim.Optimizer): Optimizador.
        num_epochs (int): Número de épocas de entrenamiento.
        train_loader (iterable): Cargador de datos de entrenamiento.
        val_loader (iterable): Cargador de datos de validación.
        device (str): Dispositivo de cómputo.

    Returns:
        nn.Module: Modelo entrenado.
    """
    for epoch in range(num_epochs):
        transformer.train()

        total_loss, total_corr = [], []
        print(f"\nEPOCH: {epoch + 1}/{num_epochs}")
        for era_num in tqdm(train_loader):
            batch = train_loader[era_num]
            loss, _mse, _corr = train_on_batch(transformer, optimizer, batch, device)
            total_loss.append(loss)
            total_corr.append(_corr)

            if np.isnan(loss):
                print("Error: pérdida NaN detectada, deteniendo el entrenamiento.")
                break

        print(f"Train Loss: {np.mean(total_loss):.4f} | Train Corr: {np.mean(total_corr):.4f}")
        train_corr = np.mean(total_corr)

        # Evaluación en conjunto de validación
        transformer.eval()
        with torch.no_grad():
            total_loss, total_corr = [], []
            for era_num in tqdm(val_loader):
                batch = val_loader[era_num]
                loss, _mse, _corr = evaluate_on_batch(transformer, batch, device)
                total_loss.append(loss)
                total_corr.append(_corr)

            print(f"Val Loss: {np.mean(total_loss):.4f} | Val Corr: {np.mean(total_corr):.4f}")
        val_corr = np.mean(total_corr)

        # Liberar caché de memoria
        torch.cuda.empty_cache()
        _ = gc.collect()

        # Condición de parada si se detecta sobreajuste extremo
        if train_corr > 4 * val_corr:
            print("Sobreajuste detectado, deteniendo el entrenamiento.")
            break

    # Guardar el estado del modelo entrenado
    torch.save(transformer.state_dict(), f"transformer_epoch_{epoch}.pth")

    return transformer