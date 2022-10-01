import numpy as np
import pandas as pd
import matplotlib as plt

# Fixar semente aleatória para poder reproduzir os mesmos resultados
semente = 1
np.random.seed(semente)
#configurar impressão de np.array
np.set_printoptions(precision=2, suppress= True)
# Número de amostras/exemplos/tamanho do conjunto de dados
amostras = 100

# Gerar dados a serem modelados
def gerar_dados_lineares(intervalo_x: tuple, coeficiente_angular: float, ruido: int, coeficiente_linear: float) -> np.array:
    x = np.random.randint(intervalo_x[0], intervalo_x[1], size=amostras)
    ruido_gerado = np.random.uniform(-ruido, ruido, size=amostras)
    y = coeficiente_angular*x + coeficiente_linear + ruido_gerado
    return np.vstack([x, y]).T

# Separar conjunto de dados em conjunto de treinamento, testes e validação
def dividir_dados(dados: np.array, teste: float = .15, validacao: float = .15) -> tuple:
    treinamento = 1 - teste - validacao
    # Embaralhar dados
    np.random.shuffle(dados)
    dados_treinamento = dados[0:int(treinamento*amostras), :]
    aux = len(dados_treinamento)
    dados_testes = dados[aux:aux+int(teste*amostras), :]
    aux += len(dados_testes)
    dados_validacao = dados[aux:, :]
    return dados_treinamento, dados_testes, dados_validacao


# Padronizar os dados (média = 0, desvio padrão = 1), chamado Z-score normalization
def padronizar_dados(dados: np.array) -> np.array:
    if len(dados) == 0:
        return dados
    media = np.mean(dados, axis=0)
    desvio_padrao = np.std(dados, axis=0)
    return (dados-media)/desvio_padrao


def despadronizar_modelo(dados_originais: np.array, modelo: tuple) -> np.array:
    media = np.mean(dados_originais, axis=0)
    desvio_padrao = np.std(dados_originais, axis=0)
    w = modelo[0] * (desvio_padrao[1]/desvio_padrao[0])
    b = modelo[1] * desvio_padrao[1] + media[1] - np.sum(w*media[0])
    return w, b


def regressao_linear(dados: np.array) -> tuple:
    x = dados[:, 0]
    y = dados[:, 1]
    # Inicializar pesos
    w = np.random.rand(1)
    b = np.zeros(1)
    # parâmetros do treinamento
    # alfa — taxa de aprendizado (controla a velocidade no ajuste dos pesos)
    alfa = 0.1
    num_epocas = 20
    # executa treinamento
    for epoca in range(num_epocas):
        y_previsto = w * x + b
        erro = erro_quadratico_medio(y_previsto, y)
        #print("Erro:", erro)
        dw, db = gradiente(y_previsto, y, x)
        # atualizar pesos (aprendizado)
        # como desejamos minimizar o erro, devemos ir no sentido contrário indicado pelo gradiente
        w += alfa * -dw
        b += alfa * -db
    return w, b


# Função de erro (MSE — Mean Squared Error)
def erro_quadratico_medio(y_previsto: np.array, y: np.array) -> float:
    return (1/y.size) * np.sum((y - y_previsto)**2)


def erro_modelo(dados: np.array, modelo: tuple) -> float:
    x = dados[:, 0]
    y = dados[:, 1]
    w = modelo[0]
    b = modelo[1]
    y_previsto = w * x + b
    return erro_quadratico_medio(y_previsto, y)


# Computar a derivada da função de erro em relação aos pesos do modelo (indica sentido para maximizar a função)
def gradiente(y_previsto: np.array, y: np.array, x: np.array) -> tuple:
    #dErro/dw
    dw = (-2/y.size) * np.sum((y - y_previsto) * x)
    #dErro/db
    db = (-2/y.size) * np.sum(y - y_previsto)
    return dw, db


def main():
    coef_ang = 5
    coef_lin = 100
    dados = gerar_dados_lineares((0, 100), coef_ang, coef_lin, 50)
    #print(dados)
    x = dados[:, 0]
    y = dados[:, 1]
    df = pd.DataFrame(dados, columns=['x', 'y'])
    #print(df)
    #plt.title("Dados gerados")
    #plt.scatter(df['x'], df['y'])
    #plt.show()

    dados_treinamento, dados_testes, dados_validacao = dividir_dados(dados, .15, 0)
    dados_treinamento_normal = padronizar_dados(dados_treinamento)
    dados_testes_normal = padronizar_dados(dados_testes)
    dados_validacao_normal = padronizar_dados(dados_validacao)

    w, b = regressao_linear(dados_treinamento_normal)
    modelo = (w, b)
    erro_treinamento = erro_modelo(dados_treinamento_normal, modelo)
    erro_testes = erro_modelo(dados_testes_normal, modelo)
    print("Erro treino:", erro_treinamento)
    print("Erro teste: ", erro_testes)

    print("Modelo Real: %.2fx + %.2f + ruído" % (coef_ang, coef_lin))
    w, b = despadronizar_modelo(dados, modelo)
    print("Modelo: %.2fx + %.2f" % (w[0], b[0]))

    # Configurar tamanho da figura
    plt.figure(figsize=(15, 5))

    df_treinamento = pd.DataFrame(dados_treinamento, columns=['x', 'y'])
    df_treinamento['y_resultado'] = df_treinamento['x'] * w + b
    df_testes = pd.DataFrame(dados_testes, columns=['x', 'y'])
    df_testes['y_resultado'] = df_testes['x'] * w + b

    plt.subplot(1, 2, 1)
    plt.title("Dados de Treinamento")
    plt.scatter(df_treinamento['x'], df_treinamento['y'], label="treinamento")
    plt.plot(df_treinamento['x'], df_treinamento['y_resultado'], color="red", linewidth=1, linestyle="-", label="modelo")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.title("Dados de Teste")
    plt.scatter(df_testes['x'], df_testes['y'], label='teste')
    plt.plot(df_testes['x'], df_testes['y_resultado'], color="red", linewidth=1, linestyle="-", label="modelo")
    plt.legend(loc="lower right")

    # Show plots
    plt.show()
main()
