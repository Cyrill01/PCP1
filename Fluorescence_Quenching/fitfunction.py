def gauss_expo_convolution(data, a, mu, sigma, tau):
    res = []
    for x in data:
        expo = np.exp((sigma**2-2*tau*x+2*mu*tau)/(2*tau**2))
        ero = math.erfc((tau*(mu-x)+sigma**2)/(np.sqrt(2)*sigma*tau))
        res.append(expo * ero)
    return a * np.array(res)
