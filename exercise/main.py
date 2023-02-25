# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np

gpu = torch.device("cuda")
cpu="cpu"
# Press the green button in the gutter to run the script.
if __name__ == '__main__':


   sequence_tensor=torch.tensor(np.array([[[1,2,3],
                                            [4,5,6]],

                                           [[9,8,7],
                                            [6,5,4]]]),
                                 dtype=torch.float,requires_grad=True,device=gpu)
   sequence_tensor_deepCp=torch.tensor(sequence_tensor.to(cpu).detach().numpy())
   sequence_tensor_deepCp+=1
   print(sequence_tensor)
   print(sequence_tensor_deepCp)

   print(sequence_tensor.var(0))












   #     fp = open("test.txt", "w+")
   #     fp.write("1111111\n")
   #     fp.write("2222222\n")
   #     fp.seek(0,0)
   #     line1=fp.readline()
   #
   #
   # except OSError :
   #     print("OSError")
   #
   #
   # # def takesecond(tuple):
   # #     return tuple[1]
   # # list=[(2,2),(3,1),(4,3), {9,8}]
   # # list2=list[4:0:-2]
   # # print(list2)
   #
   #
   #
   # def dec_fun(func):
   #     print("dec1")
   #     def internal_func():
   #         print("1")
   #         func()
   #         print("2")
   #         return
   #     return internal_func
   #
   # def dec_fun_2(func):
   #     print("dec2")
   #     def internal_func():
   #         print("a")
   #         func()
   #         print("b")
   #         return
   #     return internal_func
   #
   # @dec_fun
   # @dec_fun_2
   # def main_func():
   #     print("hello")
   #     return
   #
   # main_func()








