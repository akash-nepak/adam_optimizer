import numpy as np
import matplotlib.pyplot as plt 


def quadratic_loss (x,y): #using a simple quadratic loss function
    return x**2 + 10 * y**2


def quadratic_grad(x,y):
    dx =  2  * x
    dy = 20  * y

    return np.array([dx,dy])


def gradient_descent(grad_func,lr,epochs,start_point):
    x,y = start_point
    path = [(x,y)] #list for appending x,y path
    losses = [quadratic_loss(x,y)]

    for _ in range (epochs):
        grad = grad_func(x,y)
        x -= lr * grad[0] #access dx
        y -= lr * grad[1] #access dy 

        path.append((x,y))
        losses.append(quadratic_loss(x,y))


    return np.array(path),losses 


def adam_optimizer(grad_func ,lr,  epochs , beta_1, beta_2,epsilon,start_point):
    x , y = start_point
    m = np.array([0.0,0.0]) #momentum terms
    v = np.array([0.0,0.0]) # cached memory decay
    path = [(x,y)]
    losses = [quadratic_loss(x,y)]


    for t in range (1, epochs + 1):

        grad = grad_func(x , y)
        m = beta_1 * m + (1-beta_1) * grad
        v = beta_2 * v + (1- beta_2) * (grad ** 2)

        #corrected m ,v values (1-beta**t)
        m_hat = m / (1- beta_1 ** t)
        v_hat = v /(1 - beta_2 ** t)

        x -= lr * m_hat[0]/ (np.sqrt(v_hat[0]) + epsilon)
        y -= lr * m_hat[1]/ (np.sqrt(v_hat[1]) + epsilon)

        path.append((x,y))
        losses.append(quadratic_loss(x,y))

    return np.array(path), losses

def plot_losses(losses,labels,title):
    plt.figure(figsize = (8,6))

    for loss, label in zip(losses,labels):
        plt.plot(loss,label =label)

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


lr_gd = 0.1
lr_adam = 0.1
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
epochs = 100
start_point = (1.5,1.5)

    






