import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
data_df = pd.read_csv('data/train.csv')
X = data_df.drop('Activity', axis=1)
y = data_df['Activity']

# 1. Variance analysis
variances = X.var().sort_values(ascending=False)
variances_df = variances.reset_index()
variances_df.columns = ['Feature', 'Variance']
variances_df_top50 = variances_df.head(50)

variance_fig = px.bar(variances_df_top50, x='Feature', y='Variance', 
    title='Varianza de las 50 Caracteristicas con Mayor Valor', height=500)
variance_fig.update_xaxes(tickangle=90, title_text='')

# 2. 3D plot of top variance features
top_3_features = variances.head(3).index
top_3_df = data_df[top_3_features].copy()
top_3_df['Activity'] = y
scatter_3d_fig = px.scatter_3d(top_3_df, x=top_3_features[0], y=top_3_features[1], 
    z=top_3_features[2], color='Activity', height=700)
scatter_3d_fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# 3. PCA
X_scaled = StandardScaler().fit_transform(X)
n_components = min(X.shape[0], X.shape[1])
pca = PCA(n_components=n_components)
pca.fit(X_scaled)
latent = pca.explained_variance_

pca_variance_df = pd.DataFrame({
    'PC': [f'PC-{i+1}' for i in range(n_components)],
    'Variance': latent
})
pca_variance_df['Cumulative'] = pca_variance_df['Variance'].cumsum() / pca_variance_df['Variance'].sum()

scree_plot_fig = px.bar(pca_variance_df, x='PC', y='Variance', height=500)
scree_plot_fig.add_scatter(x=pca_variance_df['PC'], y=pca_variance_df['Cumulative'], 
    yaxis='y2', name='Cumulative', mode='lines+markers')
scree_plot_fig.update_layout(yaxis2=dict(title='Cumulative %', overlaying='y', side='right', tickformat=".0%"))

# 4. Dimensionality reduction
n_components_95 = (pca_variance_df['Cumulative'] >= 0.95).idxmax() + 1
original_dims = X.shape[1]
reduction_percentage = (1 - (n_components_95 / original_dims)) * 100

# 5. PCA 2D and 3D visualization
X_pca = pca.fit_transform(X_scaled)
pca_scores_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
pca_scores_df['Activity'] = y

scatter_2d_fig = px.scatter(pca_scores_df, x='PC1', y='PC2', color='Activity', height=600,
    title='Proyeccion 2D de los Datos en el Espacio de Componentes Principales')

scatter_3d_pca_fig = px.scatter_3d(pca_scores_df, x='PC1', y='PC2', z='PC3', color='Activity', height=700,
    title='Proyeccion 3D de los Datos en el Espacio de Componentes Principales')

# 6. Loadings
top_pcs = 3
components_df = pd.DataFrame(pca.components_[:top_pcs, :].T, 
    columns=[f'PC{i+1}' for i in range(top_pcs)], index=X.columns)

loadings_figs = []
for i in range(top_pcs):
    pc = f'PC{i+1}'
    top = components_df[pc].abs().nlargest(10).reset_index()
    top.columns = ['Feature', 'Value']
    fig = px.bar(top, x='Feature', y='Value', 
        title=f'Top 10 Features for {pc}', height=400)
    fig.update_xaxes(tickangle=45, title_text='')
    loadings_figs.append(fig)

# Create HTML
with open('report.html', 'w') as f:
    f.write(f'<html><head><title>Analisis de Varianza y PCA</title>')
    f.write('<style>body { font-family: sans-serif; margin: 2em; } .section { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px;>{{border-radius: 5px; }}</style></head><body>')
    f.write('<h1>Taller: Analisis de Varianza y Componentes Principales</h1>')
    f.write('<h2>Dataset: Human Activity Recognition with Smartphones</h2><hr>')
    
    f.write('<div class="section"><h3>Punto 1 & 2: Analisis de Varianza</h3>')
    f.write(f'<p>{X.shape[1]} caracteristicas</p>')
    f.write(pio.to_html(variance_fig, full_html=False, include_plotlyjs=True))
    f.write('</div>')
    
    f.write('<div class="section"><h3>Punto 3: Visualizacion 3D</h3>')
    f.write(f'<p>Features: {", ".join(top_3_features)}</p>')
    f.write(pio.to_html(scatter_3d_fig, full_html=False, include_plotlyjs=False))
    f.write('</div>')
    
    f.write('<div class="section"><h3>Punto 4-7: PCA</h3>')
    f.write(pio.to_html(scree_plot_fig, full_html=False, include_plotlyjs=False))
    f.write('</div>')
    
    f.write('<div class="section"><h3>Punto 8: Conclusiones</h3>')
    f.write(f'<p>Componentes para 95% varianza: {n_components_95}<br>')
    f.write(f'Original: {original_dims}<br>Reduccion: {reduction_percentage:.2f}%</p></div>')
    
    f.write('<div class="section"><h3>Visualizacion PCA 2D y 3D</h3>')
    f.write(pio.to_html(scatter_2d_fig, full_html=False, include_plotlyjs=False))
    f.write(pio.to_html(scatter_3d_pca_fig, full_html=False, include_plotlyjs=False))
    f.write('</div>')
    
    f.write('<div class="section"><h3>Analisis de Loadings</h3>')
    for fig in loadings_figs:
        f.write(pio.to_html(fig, full_html=False, include_plotlyjs=False))
    f.write('</div>')
    
    f.write('</body></html>')

print("Reporte generado exitosamente.")
