# -*- coding: utf-8 -*-

# Copyright 2021 Zhang, Chen. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# @since 2021/4/27 14:46
# @author Zhang, Chen (chansonzhang)
# @email ZhangChen.Shaanxi@gmail.com
import numpy as np

A = np.matrix([[1, 0.3], [0.45, 1.2]])
U, s, V = np.linalg.svd(A)
print("U: \n{}".format(U))
print("s: \n{}".format(s))
print("V: \n{}".format(V))

print(len(U))

# Verify calculation of A=USV
print(np.allclose(A, U * np.diag(s) * V))

# Verify orthonormal properties of U and V. (Peformed on U but the same applies for V).
#  1) Dot product between columns = 0
print(np.round([np.dot(U[:, i-1].A1,  U[:, i].A1) for i in range(1, len(U))]))


#  2) Columns are unit vectors (length = 1)
print(np.round(np.sum((U*U), 0)))

#  3) Multiplying by its transpose = identity matrix
print(U.T*U)
print(np.allclose(U.T * U, np.identity(len(U))))



