import torch
pthfile=r'./ckpt/depth.pth'
model=torch.load(pthfile,torch.device('cpu'))

print('type:')
print(type(model))  #查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  #查看模型字典里面的key
    print(k)

