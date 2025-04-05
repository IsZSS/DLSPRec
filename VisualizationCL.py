import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

<<<<<<< HEAD
# name = 'PHO_Test_Undisen_2_'
# name = 'PHO_disen_'

# name = 'NYC_Test_Undisen_2_'
# name = 'NYC_disen_'

name = 'SIN_Test_Undisen_2_'
=======
#
# name = 'NYC NoCL'
#
# preference_file = open(f"./results/{name} preference", 'rb')
# # preference_file = open(f"./results/{name} preference", 'rb')
# [embedding_weights_1, embedding_weights_2] = pickle.load(preference_file)
# preference_file.close()
#
# # 使用 sklearn 的 TSNE 计算 t-SNE 表示（降到2维）
# # tsne = TSNE(n_components=2, perplexity=30.0, n_iter=300, random_state=2)
# # tsne = TSNE(random_state=3, n_iter=1000, metric="cosine")
# tsne = TSNE(random_state=2023, perplexity=80.0, n_iter=300, metric="cosine")
# embedded_vectors_1 = tsne.fit_transform(embedding_weights_1)
# embedded_vectors_2 = tsne.fit_transform(embedding_weights_2)
#
# # 可视化 t-SNE 结果
# plt.figure(figsize=(10, 10))
# plt.scatter(embedded_vectors_1[:, 0], embedded_vectors_1[:, 1], color=(141 / 255, 141 / 255, 141 / 255), label='Long')
# plt.scatter(embedded_vectors_2[:, 0], embedded_vectors_2[:, 1], color=(156 / 255, 79 / 255, 161 / 255), label='Short')
# plt.title(f't-SNE Visualization of {name} Preference')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.show()

# name = 'PHO_Test_Undisen_2_'
name = 'PHO_disen_'

# name = 'NYC_Test_Undisen_2_'
# name = 'NYC_disen_'
# name = 'SIN_Test_Undisen_2_'
>>>>>>> 41c5ebb2608b5923fef7c3ac9737a1e70cd3e66d
# name = 'SIN_disen_'

preference_file = open(f"./results/{name}preference", 'rb')
[embedding_weights_1, embedding_weights_2] = pickle.load(preference_file)
preference_file.close()

# perplexities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# perplexities = [30]
<<<<<<< HEAD
# states = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
states = [2012, 2013, 2014, 2015]
for state in states:
    # for perplexity in perplexities:
    # 使用 sklearn 的 TSNE 计算 t-SNE 表示（降到2维）
    tsne = TSNE(random_state=state, perplexity=5, n_iter=350, metric="cosine")  # 未解耦用这个,默认5/300
    # tsne = TSNE(random_state=state, perplexity=60, n_iter=550, metric="cosine")  # 解耦时
=======
states = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

for state in states:
    # for perplexity in perplexities:
    # 使用 sklearn 的 TSNE 计算 t-SNE 表示（降到2维）
    # tsne = TSNE(random_state=state, perplexity=30.0, n_iter=290, metric="cosine")  # 默认 未解耦用这个
    tsne = TSNE(random_state=state, perplexity=40, n_iter=500, metric="cosine")  # 解耦时暂定n_iter=500
>>>>>>> 41c5ebb2608b5923fef7c3ac9737a1e70cd3e66d

    embedded_vectors_1 = tsne.fit_transform(embedding_weights_1)
    embedded_vectors_2 = tsne.fit_transform(embedding_weights_2)
    # embedded_vectors_3 = tsne.fit_transform(embedding_weights_3)

    # 可视化 t-SNE 结果
<<<<<<< HEAD
    plt.figure(figsize=(24, 24))
    plt.scatter(embedded_vectors_1[:, 0], embedded_vectors_1[:, 1], color=(149 / 255, 213 / 255, 184 / 255),
                label='Long')  # 绿色
    plt.scatter(embedded_vectors_2[:, 0], embedded_vectors_2[:, 1], color=(104 / 255, 172 / 255, 213 / 255),
                label='Short')  # 蓝色
=======
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded_vectors_1[:, 0], embedded_vectors_1[:, 1], color=(141 / 255, 141 / 255, 141 / 255),
                label='Long')
    plt.scatter(embedded_vectors_2[:, 0], embedded_vectors_2[:, 1], color=(156 / 255, 79 / 255, 161 / 255),
                label='Short')

>>>>>>> 41c5ebb2608b5923fef7c3ac9737a1e70cd3e66d
    plt.title(f't-SNE Visualization of {name} Preference')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    # save plot to file
<<<<<<< HEAD
    # plt.savefig(f'./results/{name} preference.png', dpi=300)
=======
    # plt.savefig(f'./results/{name} preference perplexity {perplexity}.png', dpi=300)
>>>>>>> 41c5ebb2608b5923fef7c3ac9737a1e70cd3e66d
