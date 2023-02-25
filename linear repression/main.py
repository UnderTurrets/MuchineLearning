if __name__=="__main__":

    import torch
    import numpy as np
    from torch import nn
    from torch import optim
    gpu=torch.device("cuda")


    # x=torch.ones(5,3,requires_grad=True)
    # print(x)
    #
    #
    #
    #
    # y=torch.empty(4,4,requires_grad=True,device=gpu)
    # z=torch.tensor([[1,23,4,6,6],[1,1,1,1,1]],dtype=torch.float32,requires_grad=True,device=gpu)
    # a=x
    #
    #
    #
    #
    #
    # c=torch.tensor(x.detach().numpy())
    # c+=1
    # print("x",x,x.requires_grad,x.dtype)
    # print("c",c)

    x_train=torch.rand([500,2],requires_grad=True,device=gpu,dtype=torch.float32)
    x_train.data[:,1]=1
    theta_true=torch.tensor([[19.557],[-0.9871]],requires_grad=True,
                            device=gpu,dtype=torch.float32)
    y_train=torch.matmul(x_train,theta_true)


    theta=torch.randn([2,1],requires_grad=True,device=gpu,dtype=torch.float32)



    for i in range(1500):
        y_predict = torch.matmul(x_train, theta)
        cost = (y_train - y_predict).pow(2).mean()
        if theta.grad is not None:
            theta.grad.zero_()
        cost.backward(retain_graph=True)
        theta.data-=0.03*theta.grad
        if i%10==0:
            print("theta",theta)
            print(cost.item())



    class lr(nn.Module):
        def __init__(self):
            super(lr,self).__init__()
            self.linear=nn.Linear(1,1)

        def forward(self,x):
            y_predict=self.linear(x)
            return y_predict


    x_train=torch.rand([500,1],device=gpu)
    y_train=torch.matmul(x_train,torch.tensor([[3]],dtype=torch.float32,requires_grad=True,device=gpu))+8


    model_lr=lr().to(gpu)
    optimizer=optim.SGD(model_lr.parameters(),0.02)
    cost_fn=nn.MSELoss()

    for i in range(1000):
         y_predict=model_lr.linear(x_train)
         cost=cost_fn(y_predict,y_train)
         optimizer.zero_grad()

         cost.backward(retain_graph=True)

         optimizer.step()

         if i%20==0:
             print(cost.item())
             print(list(model_lr.parameters()))






