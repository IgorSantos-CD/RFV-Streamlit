import numpy                as np 
import pandas               as pd
import streamlit            as st
import matplotlib.pyplot    as plt

from datetime               import datetime
from io                     import BytesIO
from PIL                    import Image
from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans


#Set tema do seaborn para melhorar o visual dos plots
custom_params = {"axes.spines.right": False, "axes.spines.top": False}

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    writer.close()
    processed_data = output.getvalue()
    return processed_data

### Criando segmentos

def recencia_class(x, r, q_dict):
    """Classifica como melhor o menor quartil 
       x = valor da linha,
       r = recencia,
       q_dict = quartil dicionario   
    """
    if x <= q_dict[r][0.25]:
        return 'A'
    elif x <= q_dict[r][0.50]:
        return 'B'
    elif x <= q_dict[r][0.75]:
        return 'C'
    else:
        return 'D'


def freq_val_class(x, fv, q_dict):
    """Classifica como melhor o maior quartil 
       x = valor da linha,
       fv = frequencia ou valor,
       q_dict = quartil dicionario   
    """
    if x <= q_dict[fv][0.25]:
        return 'D'
    elif x <= q_dict[fv][0.50]:
        return 'C'
    elif x <= q_dict[fv][0.75]:
        return 'B'
    else:
        return 'A'
    

### Função Principal da Aplicação
def main():
    #Configuração inicial da página da aplicação
    st.set_page_config(page_title="RFV", \
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #Titulo da Aplicação
    st.write("""# RFV
             
    **RFV** significa recência, frequência, valor e é utilizado para segmentação de clientes baseado no comportamento de compras dos clientes e agrupa eles em clusters parecidos. Utilizando esse tipo de agrupamento podemos realizar ações de marketing e CRM melhores direcionadas, ajudando assim na personalização do conteúdo e até a retenção de clientes.

    Para cada cliente é preciso calcular cada uma das componentes abaixo:

    - Recência (R): Quantidade de dias desde a última compra.
    - Frequência (F): Quantidade total de compras no período.
    - Valor (V): Total de dinheiro gasto nas compras do período.

    E é isso que iremos fazer abaixo.
    """)
    st.markdown("---")

    #Botão para carregar o arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Marketing Data", type=["csv", "xlsx"])

    #Vefica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        
        df_compras = pd.read_csv(data_file_1,infer_datetime_format=True,parse_dates=['DiaCompra'])

        st.write("## Recência (R)")

        dia_atual = df_compras['DiaCompra'].max()
        st.write("Dia máximo na base de dados:", dia_atual)

        st.write("Quantos dias faz que o cliente fez sua ultima compra?")

        #group by customers and check last date of purshace
        df_recencia = df_compras.groupby(by='ID_cliente', as_index=False)['DiaCompra'].max()
        df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']
        df_recencia['Recencia'] = df_recencia['DiaUltimaCompra'].apply(lambda x: (dia_atual - x).days)
        st.write(df_recencia.head())

        df_recencia.drop('DiaUltimaCompra', axis=1, inplace=True)

        ## Frequência
        st.write("## Frequência (R)")
        st.write("Quantas vezes cada cliente comprou com a gente?")
        df_frequencia = df_compras[['ID_cliente', 'CodigoCompra']].groupby('ID_cliente').count().reset_index()
        df_frequencia.columns = ['ID_cliente', 'Frequencia']
        st.write(df_frequencia.head())

        ## Valor
        st.write("## Valor (V)")
        st.write("Quanto que cada cliente gastou no periodo?")
        df_valor = df_compras[['ID_cliente', 'ValorTotal']].groupby('ID_cliente').sum().reset_index()
        df_valor.columns = ['ID_cliente', 'Valor']
        st.write(df_valor.head())

        st.write("## Tabela RFV Final")
        df_RF = df_recencia.merge(df_frequencia, on='ID_cliente')
        df_RFV = df_RF.merge(df_valor, on='ID_cliente')
        df_RFV.set_index('ID_cliente', inplace=True)
        st.write(df_RFV.head())

        st.write("## Segmentação de clientes utilizando o RFV")
        st.write("Para realizar esta segmentação, podemos utilizar o modelo não supervisonado K-MEANS para agrupar os clientes onde as caracterisiticas se adequem conversem melhor. Abaixo esta a utilização de agrupamento utilizando K-MEANS")

        # --- Padronizar variáveis RFV ---
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_RFV[["Recencia", "Frequencia", "Valor"]])

        inertia = []
        K = range(2, 10)
        for k in K:
            km = KMeans(n_clusters=k, random_state=3326)
            km.fit(df_scaled)
            inertia.append(km.inertia_)

        st.write("### Método do Cotovelo (Inércia x K)")
        fig, ax = plt.subplots()
        fig.set_figwidth(25.5)
        ax.plot(K, inertia, marker="o")
        ax.set_xlabel("N° de Clusters (k)")
        ax.set_ylabel("Inércia")
        st.pyplot(fig)

        # --- Aplciando o K-means com o número de clusters ideal ---

        k = 5
        kmeans = KMeans(n_clusters=k, random_state=3327)
        df_RFV["Cluster"] = kmeans.fit_predict(df_scaled)

        st.write("### Resumo dos clusters K-means")
        st.write(df_RFV.groupby("Cluster")[['Recencia','Frequencia','Valor']].mean())

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df_RFV['Recencia'],
            df_RFV['Valor'],
            c=df_RFV['Cluster'],  # cores por cluster
            alpha=0.6
        )

        # Pega o colormap que o scatter usou
        cmap = scatter.cmap
        norm = scatter.norm

        handles = []
        labels = []

        for cluster_label in sorted(df_RFV['Cluster'].unique()):
            color = cmap(norm(cluster_label))  # cor exata do cluster
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=10)
            )
            labels.append(f"Cluster {cluster_label}")

        ax.legend(handles, labels, title="Clusters")

        fig.set_figwidth(25.5)
        ax.set_xlabel("Recência")
        ax.set_ylabel("Valor")
        ax.set_title("Clusters K-means (Recência x Valor)")

        st.pyplot(fig)

        st.write("Podemos observar que o agrupamento nos resultou em 5 grupos até que bem segmentados, onde poderiamos utilizar como os melhores grupos de clientes, considerando Recência e Valor Gasto, os Clusters 2 e 4")

        st.write("Outro jeito de segmentar os clientes é criando quartis para cada componente do RFV, sendo que o melhor quartil é chamado de 'A', o segundo melhor quartil de 'B', o terceiro melhor de 'C' e o pior de 'D'. O melhor e o pior depende da componente. Po exemplo, quanto menor a recência melhor é o cliente (pois ele comprou com a gente tem pouco tempo) logo o menor quartil seria classificado como 'A', já pra componente frêquencia a lógica se inverte, ou seja, quanto maior a frêquencia do cliente comprar com a gente, melhor ele/a é, logo, o maior quartil recebe a letra 'A'.")
        st.write("Se a gente tiver interessado em mais ou menos classes, basta a gente aumentar ou diminuir o número de quantils pra cada componente.")

        st.write("Quartis para o RFV")

        quartis = df_RFV.quantile(q=[0.25, 0.5, 0.75])
        st.write(quartis)

        df_RFV['R_quartil'] = df_RFV['Recencia'].apply(recencia_class,
                                                args=('Recencia', quartis))
        df_RFV['F_quartil'] = df_RFV['Frequencia'].apply(freq_val_class,
                                                  args=('Frequencia', quartis))
        df_RFV['V_quartil'] = df_RFV['Valor'].apply(freq_val_class,
                                             args=('Valor', quartis))

        df_RFV['RFV_Score'] = (df_RFV.R_quartil 
                            + df_RFV.F_quartil 
                            + df_RFV.V_quartil)
        st.write(df_RFV.head())

        st.write("Quantidade de Clientes por Grupo")
        st.write(df_RFV['RFV_Score'].value_counts())

        st.write("#### Cliente com menor recência, maior frequencia e maior valor gasto")
        st.write(df_RFV[df_RFV['RFV_Score'] == 'AAA'].sort_values('Valor',ascending=False).head(10))


        st.write("### Ações de marketing/CRM")

        dict_acoes = {'AAA':'Enviar cupons de desconto, Pedir para indicar nosso produto pra algum amigo, Ao lançar um novo produto enviar amostras grátis pra esses.',
        'DDD':'Churn! clientes que gastaram bem pouco e fizeram poucas compras, fazer nada',
        'DAA':'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar',
        'CAA':'Churn! clientes que gastaram bastante e fizeram muitas compras, enviar cupons de desconto para tentar recuperar'
        }

        df_RFV['acoes de marketing/crm'] = df_RFV['RFV_Score'].map(dict_acoes)
        st.write(df_RFV.head())

        #df_RFV.to_excel('./output/RFV_.xlsx')
        df_xlsx = to_excel(df_RFV)
        st.download_button(label="Download",
                           data=df_xlsx,
                           file_name="RFV_.xlsx")
        
        st.write("### Quantidade de clientes por tipo de ação")
        st.write(df_RFV["acoes de marketing/crm"].value_counts(dropna=False))

if __name__ == "__main__":
    main()