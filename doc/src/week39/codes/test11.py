##Make synthetic data
n = 1000  
np.random.seed(20)
x1 = np.random.rand(n)
x2 = np.random.rand(n)     
X = designMatrix(x1, x2, 4)
y = franke(x1, x2) 

##Train-validation-test samples. 
# We choose / play with hyper-parameters on the validation data and then test predictions on the test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

X_train[:, 0] = 1
X_test[:, 0] = 1
X_val[:, 0] = 1



linreg = linregOwn(method='ols')
#print('Invert OLS:', linreg.fit(X_train, y_train))
beta = SGD(X_train, y_train, learning_rate=0.07)
#print('SGD OLS:', beta)


linreg = linregOwn(method='ridge')
#print('Invert Ridge:', linreg.fit(X_train, y_train, lambda_= 0.01))
beta = SGD(X_train, y_train, learning_rate=0.0004, method='ridge')
#print('SGD Ridge:', beta)


sgdreg = SGDRegressor(max_iter = 100, penalty=None, eta0=0.1)
sgdreg.fit(X_train[:, 1:],y_train.ravel())
#print('sklearn:', sgdreg.coef_)
#print('sklearn intercept:', sgdreg.intercept_)


def plot_MSE(method = 'ridge', scheme = None):
    eta = np.logspace(-5, -3, 10)
    lambda_ = np.logspace(-5, -1, 10)
    MSE_ols = []
    MSE_ridge = []
    
    if scheme == 'joint':
        
        if method == 'ridge':
            
            for lmbd in lambda_:
                
                for i in eta:  
                    
                    beta = SGD(X_train, y_train, learning_rate=i, lambda_ = lmbd, method = method)
                    mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, lambda_ = lmbd, beta = beta)
                    MSE_ridge.append(mse_ridge_test)
            
            fig = plt.figure()
            ax = fig.gca(projection='3d') ##get current axis
            lambda_ = np.ravel(lambda_)
            eta = np.ravel(eta)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.03f'))
            ax.plot_trisurf(lambda_, eta, MSE_ridge, cmap='viridis', edgecolor='none')
            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(r'$\eta$')
            ax.set_title(r'MSE Ridge')
            ax.view_init(30, 60)
            plt.show()
      
    if scheme == 'separate':
        
        if method == 'ols':
            
            eta = np.logspace(-5, 0, 10)  
            
            for i in eta:  
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = method)
                mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, beta = beta)
                MSE_ols.append(mse_ols_test)
            
            print('The learning rate {} performs best for the OLS' .format(eta[MSE_ols.index(min(MSE_ols))]))
            print('Corresponding minimum MSE for OLS: {}'.format(min(MSE_ols)))
            plt.semilogx(eta, MSE_ols)
            plt.xlabel(r'Learning rate, $\eta$')
            plt.ylabel('MSE OLS')
            plt.title('Stochastic Gradient Descent')
            plt.show()
    if scheme == 'separate':
        
        if method == 'ridge':
            
            eta = np.logspace(-5, 0, 10)  
            
            for i in eta:  
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = method)
                mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, beta = beta)
                MSE_ols.append(mse_ridge_test)
            
            print('The learning rate {} performs best for Ridge' .format(eta[MSE_ols.index(min(MSE_ols))]))
            print('Corresponding minimum MSE for Ridge: {}'.format(min(MSE_ols)))

            plt.plot(eta, MSE_ols)
            plt.xlabel(r'Learning rate, $\eta$')
            plt.ylabel('MSE Ridge')
            plt.title('Stochastic Gradient Descent')
            plt.show()
        

 
# plot_MSE(method='ridge', scheme = 'joint')

# plot_MSE(method='ols', scheme = 'separate')

# plot_MSE(method='ridge', scheme = 'separate')


####Predict OLS, Ridge on test data after tuning learning rate and lambda on validation data

def plot_scatter(y_true, method = 'ols'):
    if method == 'ols':
        beta = SGD(X_train, y_train, learning_rate=0.07, lambda_ = 0, method = method, n_epochs=300)
    if method == 'ridge':
        beta = SGD(X_train, y_train, learning_rate=0.0001, lambda_ = 0, method = method, n_epochs=300)
    y_pred = np.dot(X_test, beta)
    mse_ols_test, mse_ridge_test = compute_test_mse(X_test, y_true, beta = beta)
    print('Test MSE OLS: {}' .format(mse_ols_test))
    print('Test MSE Ridge: {}' .format(mse_ridge_test))
    a = plt.axes(aspect='equal')
    plt.scatter(y_pred, y_pred, color= 'blue', label = "True values")
    plt.scatter(y_pred, y_true, color = 'red', label = "Predicted values")
    plt.xlabel('True y values')
    plt.ylabel('Predicted y')
    plt.title(f"Prediction - {method}")
    plt.legend()
    # if method == 'ols':
    #     plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'ols_reg_pred.png'), transparent=True, bbox_inches='tight')
    # if method == 'ridge':
    #     plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'ridge_reg_pred.png'), transparent=True, bbox_inches='tight')
    
    plt.show()

plot_scatter(y_test, method='ols')

plot_scatter(y_test, method='ridge')
