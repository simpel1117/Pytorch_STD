# class Foo(object):
#     def __init__(self):
#         self.price =100
#         self.discount =0.8
    
   
#     def  get_price(self):
#         new_price = self.price*self.discount
#         return new_price
    
#     def set_price(self, value):
#         self.price =value
    
#     def del_price(self):
#         del self.price


#     PRICE =property(get_price, set_price, del_price ,"价格展示...")


# obj = Foo()
# print(obj.PRICE)
# print("okkkk")
# obj.price =200
# print(obj.PRICE)



# class Goods(object):
#     def __init__(self):
#         self.original_price = 100
#         self.discount = 0.8
    
#     @property
#     def price(self):
#         new_price = self.original_price * self.discount
#         return new_price
    
#     @price.setter
#     def price(self, value):
#         self.original_price = value
    
#     @price.deleter
#     def price(self):
#         del self.original_price

# obj = Goods()
# print(obj.price)
# obj.price = 200
# print(obj.price)


# class Money(object):
#     def __init__(self):
#         self.__money = 0
    
#     def getMoney(self):
#         return self.__money
    
#     def setMoney(self, value):
#         if isinstance(value, int):
#             self.__money = value
#         else:
#             print("error: 不是整数")

#     money = property(getMoney,setMoney)

# a = Money()
# a.money = 100
# print(a.money)        

class Money(object):
    def __init__(self):
        self.__money = 0
    
    @property
    def money(self):
        return self.__money
    
    @money.setter
    def money(self, value):
        if isinstance(value, int):
            self.__money = value
        else:
            print("wrong")

a = Money()
a.money = 100
print(a.money)  