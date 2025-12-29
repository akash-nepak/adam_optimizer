import numpy as np

class Optimizer_Adam:

    def __init__(self,learning_rate = 0.001, decay = 0.00, epsilon =1e-7,beta_1 = 0.9, beta_2 = 0.999): #parametrs needed for adam optimizer
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay #using decaying learning rate 
        self.iterations = 0
        self.epsilon =epsilon
        self.beta_1 = beta_1 #importance of past gradient
        self.beta_2 = beta_2 #importance of gradient  squared


    def pre_update_params(self): #calling to change learning rate as epoch increases 

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0/ (1.0 + (self.decay * self.iterations)))

    def update_params(self,layer):

        if not hasattr(layer, 'weight_cache'): #checking if each weight of the layer has weight attribute assosiated with it

            layer.weight_momentums = np.zeros_like(layer.weight) # each weight has a momentum anD A  cache term
            layer.weight_cache = np.zeros_like(layer.weight)
            layer.bias_momentums = np.zeros_like(layer.biase)
            layer.bias_cache = np.zeros_like(layer.biase)

        #updating momentum with gradients

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1- self.beta_1)*layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1- self.beta_1)*layer.dbiases
        
        #getting corrected momentum

        weight_momentums_corrected = layer.weight_momentums / (1-(self.beta_1 ** (self.iterations +1)))
        bias_momentums_corrected = layer.bias_momentums / (1-(self.beta_1 ** (self.iterations +1)))

        #updating cache memory decay
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2

        #corrected memory cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations +1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations +1))
               
         
        layer.weight += -self.current_learning_rate * weight_momentums_corrected / np.sqrt(weight_cache_corrected + self.epsilon)
        layer.bias += -self.current_learning_rate * bias_momentums_corrected / np.sqrt(bias_cache_corrected + self.epsilon)
        

    def post_update_params(self):
        self.iterations += 1
        





        


