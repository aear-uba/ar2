import torch
import torch.nn as nn
import torch.nn.functional as F

# Hiperparámetros
state_dim = 4        # 
action_dim = 2       # 

# Capas compartidas
fc1 = nn.Linear(state_dim, 128)

# Stream de valor V(s)
fc_value1 = nn.Linear(128, 64)
fc_value2 = nn.Linear(64, 1)

# Stream de ventaja A(s,a)
fc_adv1 = nn.Linear(128, 64)
fc_adv2 = nn.Linear(64, action_dim)

# Función para hacer un forward
def dueling_forward(state):
    x = F.relu(fc1(state))                          # Capa compartida

    # Valor
    v = F.relu(fc_value1(x))
    v = fc_value2(v)                                # [batch_size, 1]

    # Ventaja
    a = F.relu(fc_adv1(x))
    a = fc_adv2(a)                                  # [batch_size, action_dim]

    # Normalización (restar la media)
    a_mean = a.mean(dim=1, keepdim=True)            # [batch_size, 1]
    q = v + (a - a_mean)                            # [batch_size, action_dim]

    return q
