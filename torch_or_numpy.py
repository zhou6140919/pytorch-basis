import torch
import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2numpy = torch_data.numpy()
#
# print(
#     'numpy_data\n', np_data,
#     'torch_data\n', torch_data,
#     'tensor2numpy\n', tensor2numpy,
# )

from torch.autograd import Variable

tensor = torch.FloatTensor(([1, 2], [3, 4]))
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor * tensor)   #x^2
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

v_out.backward()
#v_out = 1/4 * sum(var*var)
#dv_out / dvar = 1/4 * sum(var*var) = var/2
print('梯度', variable.grad)
print(variable.data)






