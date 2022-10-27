import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


def plot_explained_variance(data_pca):
    prop_varianza_acum = data_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.plot(
        np.arange(data_pca.n_components_) + 1,
        prop_varianza_acum,
        marker='o'
    )

    for x, y in zip(np.arange(data_pca.n_components_) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 1),
            ha='center'
        )

    ax.set_ylim(0.1, 1.1)
    ax.set_xticks(np.arange(data_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada')

def get_n_components_by_cum_variance(data_pca,required_ratio):
    prop_varianza_acum = data_pca.explained_variance_ratio_.cumsum()
    return prop_varianza_acum[prop_varianza_acum < required_ratio].size


# load data
digits = datasets.load_digits()
plot_digits(digits.data[:100, :])
# scale
digits_scaled = StandardScaler().fit_transform(digits.data)

# PCA

pca = PCA()
digits_pca = pca.fit(digits_scaled)
plot_explained_variance(digits_pca)
n_components = get_n_components_by_cum_variance(digits_pca,0.90)
pca = PCA(n_components=n_components)
digits_pca = pca.fit_transform(digits_scaled)
print(n_components)

#GM

bics = {}

for n in range(1, 10):
    gm = GaussianMixture(n_components=n, random_state=1)
    gm.fit(digits_pca)
    bic = gm.bic(digits_pca)
    bics[bic] = n

best_n_gm= bics[min(bics)]

print("bic",min(bics),"number of gaussian",best_n_gm)

gm = GaussianMixture(n_components=best_n_gm, random_state=1)
gm.fit(digits_pca)
gm_samples = gm.sample(100)

new_samples = pca.inverse_transform(gm_samples[0])
plot_digits(new_samples)

plt.show()
print(digits.data.shape)


