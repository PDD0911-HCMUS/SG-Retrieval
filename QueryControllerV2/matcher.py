import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class SGMatcher(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def jaccard_similarity(self, set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0.0
        return len(intersection) / len(union)
    
    @torch.no_grad()
    def forward(self, set_sg):
        N = len(data)
        score_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                score_matrix[i][j] = jaccard_similarity(data[i], data[j])

        cost_matrix = 1 - score_matrix 
        np.fill_diagonal(cost_matrix, np.inf)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return row_ind, col_ind


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0  # Tránh chia cho 0
    return len(intersection) / len(union)


data = [
    ['apple', 'banana', 'cherry'],
    ['banana', 'date', 'fig', 'grape'],
    ['cherry', 'date', 'apple'],
    ['fig', 'grape', 'melon', 'banana']
]
filenames = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']
data = [set(item) for item in data]

# Input mới
new_input = ['apple', 'date']
new_input_set = set(new_input)

# Khởi tạo ma trận điểm số
N = len(data)
score_matrix = np.zeros((N, N))

# Điền giá trị vào ma trận
for i in range(N):
    for j in range(N):
        score_matrix[i][j] = jaccard_similarity(data[i], data[j])

print(score_matrix)

cost_matrix = 1 - score_matrix 
np.fill_diagonal(cost_matrix, np.inf)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

print("Các cặp tối ưu:")
for row, col in zip(row_ind, col_ind):
    if np.isinf(cost_matrix[row, col]):
        continue
    print(f"Tập {row} được gán với Tập {col} với điểm số tương đồng {score_matrix[row, col]}")

# Kết hợp data sets với filenames
combined = list(zip(data, filenames))

similarities = [(filename, jaccard_similarity(new_input_set, data_set), data_set) for data_set, filename in combined]

top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

# In kết quả top 3
print("Top 3 Jaccard Similarities with filenames:")
for filename, similarity, da in top_similarities:
    print(f'Filename: {filename}, Jaccard Similarity: {similarity:.4f}, value: {da}')
