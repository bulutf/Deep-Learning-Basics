
#%%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
x = np.linspace(0,3,30)
y = 2*x + 3 + np.random.randn(x.size)

plt.plot(x,y,"*")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)


#%%
# Problem 1

def predict_first(x,beta0,beta1):
    
    yhat=beta0+beta1*x
    
    return(yhat)

# Problem 1 example
beta0 = 1
beta1 = 2
yhat = predict_first(x,beta0,beta1)
print(yhat[5])

#%%
# Problem 2.a 
X = np.stack((np.ones(x.shape), x), axis=1)

# Problem 2.a example
print(X[3,:])

def predict(X,beta):
    yhat=np.dot(X,beta)
    return(yhat)

# Problem 2.b example
np.random.seed(42)
beta = np.random.random(X.shape[1])
yhat = predict(X,beta)
print(yhat[5])    

#%%
# Problem 3 

def mse(y,ypred):
    sum=0;
    for i in range(yhat.shape[0]):
        diff=(y[i]-yhat[i])**2
        sum=sum+diff
    return((1/yhat.shape[0])*sum)

def mae(y,ypred):
    sum=0;
    for i in range(yhat.shape[0]):
        diff=abs(y[i]-yhat[i])
        sum=sum+diff
    return((1/yhat.shape[0])*sum)
        
def mape(y,ypred):
    sum=0;
    for i in range(yhat.shape[0]):
        diff=abs(y[i]-yhat[i])/y[i]
        sum=sum+diff
    return((1/yhat.shape[0])*sum*100)

# Problem 3 example

np.random.seed(42)
beta = np.random.random(X.shape[1])
yhat = predict(X,beta)

print("mse :", mse(y,yhat))
print("mae :", mae(y,yhat))
print("mape :", mape(y,yhat))

#%%
# Problem 4

def gradient_beta(X,y,beta):
    
    yhat=np.dot(X,beta)
    gradients=[]
    
    for k in range(X.shape[1]):
        sum=0
        diff=0

        for i in range(yhat.shape[0]):
            diff=(yhat[i]-y[i])*(1/yhat.shape[0])*X[i][k]
            sum=sum+diff
        gradients.append(sum)
          
    return(gradients)


# Problem 4 example
np.random.seed(42)
beta = np.random.random(X.shape[1])

print("Gradients are :", gradient_beta(X,y,beta))

#%%
# Problem 5 

def update_weights(beta, gradient_beta, alpha):
    
    beta_new=beta-alpha*gradient_beta

    return(beta_new)

# Problem 5 example
np.random.seed(42)
beta = np.random.random(X.shape[1])
dbeta = np.random.random(X.shape[1])*3
alpha = 0.1

print("Updated Weights are :", update_weights(beta, dbeta, alpha))

#%%
# Problem 6 

def model_fit(X,y,beta, alpha, max_iter):
    
    for x in range(max_iter):
        
        yhat=np.dot(X,beta)
        gradients=[]

        for k in range(X.shape[1]):
            sum=0
            diff=0

            for i in range(yhat.shape[0]):
                diff=(yhat[i]-y[i])*(1/yhat.shape[0])*X[i][k]
                sum=sum+diff
            gradients.append(sum)
        beta=beta-np.dot(alpha,gradients)

    return(beta)


# Problem 6 example
np.random.seed(42)
beta_init = np.random.random(X.shape[1])
alpha = 0.1
max_iter = 100
print("Coefficients are:", model_fit(X,y,beta_init, alpha, max_iter))

#%%
# Problem 7

def model_fit_ols(X,y):

    beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    
    return(beta)

# Problem 7 example

print("Coefficients are:", model_fit_ols(X,y))
