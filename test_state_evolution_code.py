from state_evolution import *

if __name__ == "__main__":

    m = 0.5
    q = 0.5
    sigma = 0.5
    rho_w_star = 1
    int_lims = 10
    alpha = 5
    epsilon = 0
    tau = 0
    Vstar = rho_w_star - m**2/q + tau**2
    e = m * m / (rho_w_star * q)
    V_0 = rho_w_star * (1-e)
    # print(Vstar,V_0)

    """
    Diggingin sigma_hat
    """
    # Vhat_x #
    def f_Vhat_plus(ξ, M, Q, V, Vstar):
        ω = np.sqrt(Q)*ξ
        ωstar = (M/np.sqrt(Q))*ξ
        λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
        return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2))) * (1 + erf(ωstar/np.sqrt(2*Vstar)))

    def f_Vhat_minus(ξ, M, Q, V, Vstar):
        ω = np.sqrt(Q)*ξ
        ωstar = (M/np.sqrt(Q))*ξ
        λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
        return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2))) * (1 - erf(ωstar/np.sqrt(2*Vstar)))
        
    def integrate_for_Vhat(M, Q, V, Vstar):
        I1 = quad(lambda ξ: f_Vhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
        I2 = quad(lambda ξ: f_Vhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
        return (1/2) * (I1 + I2)

    
    def integrand_plus(xi):
        # todo: simplify
        
        z_0 = erfc( - ( m / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,1,sigma+q,epsilon,w)

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(1,proximal,sigma+q,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal - 1)

        return z_0 * ( derivative_f_out + epsilon * (1/np.sqrt(sigma+q) - w )  ) * gaussian(xi)


    # Iplus = quad(lambda xi: integrand_plus(xi),-int_lims,int_lims,limit=500)[0]
    # Iminus = quad(lambda xi: integrand_minus(xi),-int_lims,int_lims,limit=500)[0]

    # return -alpha * 0.5 * (Iplus + Iminus)
    
    for xi in range(-10,10):
        clarte_plus = alpha * ((1/sigma) - (1/sigma**2) *  ( 0.5 * f_Vhat_plus(xi, m, q, sigma, Vstar) * gaussian(xi)) )
        mine_plus = - alpha * 0.5 * integrand_plus(xi)
        print("clarte_plus: ", clarte_plus, "mine_plus: ", mine_plus)


        ω = np.sqrt(q)*xi
        ωstar = (m/np.sqrt(q))*xi
        clarte_prox = minimize_scalar(lambda x: moreau_loss(x, 1, ω, sigma))['x']
        
        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,1,sigma+q,epsilon,w)
        print("clarte_prox: ", clarte_prox, "my proximal: ", proximal)

        clarte_z_0 = (1 + erf(ωstar/np.sqrt(2*Vstar)))
        # clarte_f_hat = (1/(1/sigma + (1/4) * (1/np.cosh(clarte_prox/2)**2))) * clarte_z_0

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(1,proximal,sigma+q,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal)  
        z_0 = erfc( - ( m / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))
        z_0_erf = 1 + erf( ( m / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))
        print("clarte_z_0: ", clarte_z_0, "my z_0: ", z_0, "my z_0_erf: ", z_0_erf)

        my_arg_z_0 = ( m * xi/ np.sqrt(q)) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q)))
        clarte_arg_z_0 = ωstar/np.sqrt(2*Vstar)
        print("my_arg_z_0: ", my_arg_z_0, "clarte_arg_z_0: ", clarte_arg_z_0)

        clarte_deriv_partial = (1/(1/sigma + (1/4) * (1/np.cosh(clarte_prox/2)**2))) * (1/sigma**2)
        my_deriv_partial = derivative_f_out
        print("clarte_deriv_partial: ", clarte_deriv_partial, "my_deriv_partial: ", my_deriv_partial)
        print("derivative_proximal", derivative_proximal, "clarte derivative proximal",(1/(1 + sigma*(1/(2*np.cosh(clarte_prox/2)**2))) ))


    m_hat = m_hat_func(m, q, sigma,rho_w_star,alpha,epsilon,tau,int_lims)
    q_hat = q_hat_func(m, q, sigma, rho_w_star,alpha,epsilon,tau,int_lims)
    sigma_hat = sigma_hat_func(m, q, sigma,rho_w_star,alpha,tau,epsilon,int_lims)
    m_hat_clart, q_hat_clart, sigma_hat_clart = update_clart(m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)
    print(f"m_hat: {m_hat}, q_hat: {q_hat}, sigma_hat: {sigma_hat}")
    print(f"m_hat_clart: {m_hat_clart}, q_hat_clart: {q_hat_clart}, sigma_hat_clart: {sigma_hat_clart}")
