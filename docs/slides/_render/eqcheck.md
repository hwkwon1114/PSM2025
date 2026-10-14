1.

$$\boldsymbol{\theta}=\left(g_{\mathrm{top}},\,g_{\mathrm{bot}},\,o_{\mathrm{top}},\,o_{\mathrm{bot}}\right)$$

2.

$$x_{g}=\frac{\ln g-\ln g_{\min}}{\ln g_{\max}-\ln g_{\min}},\qquad x_{o}=\frac{o+1}{2}$$

3.

$$q_{n}=e^{-\beta n},\quad \beta=\ln 1.25,\qquad g_{1,2}^{\ell}=q_{n}\,g_{\ell}\left(1\pm o_{\ell}\right),\quad \ell\in\{\mathrm{top},\mathrm{bot}\}$$

4.

$$\bar{\mathbf{a}}_{\ell}\leftarrow\mathbf{T}^{\mathsf{T}}\bar{\mathbf{a}}_{\ell}\,\mathbf{T},\quad \mathbf{T}=\mathbf{D}_{m}^{-1}\mathbf{R}_{\phi}\,\mathrm{diag}\left(1+g_{1}^{\ell},\,1+g_{2}^{\ell}\right)\mathbf{R}_{\phi}^{\mathsf{T}}\mathbf{D}_{m}$$

5.

$$\mathbf{x}^{*}(\boldsymbol{\theta})=\arg\min_{\mathbf{x}}\,E_{\mathrm{bilayer}}\left(\mathbf{x};\,\bar{\mathbf{a}}_{\mathrm{top}},\,\bar{\mathbf{a}}_{\mathrm{bot}}\right)$$

6.

$$\mathbf{R}=\arg\min_{\mathbf{R}\in SO(3)}\sum_{j}\left\|\mathbf{R}\left(\mathbf{p}_{j}-\bar{\mathbf{p}}\right)-\left(\mathbf{X}_{j}-\bar{\mathbf{X}}\right)\right\|^{2}$$

7.

$$L_{\mathrm{sym}}(\boldsymbol{\theta})=\sqrt{\frac{1}{2}\left[\frac{1}{N}\sum_{i=1}^{N}d\left(\mathbf{y}_{i},S_{\boldsymbol{\theta}}\right)^{2}+\sum_{j}w_{j}\min_{i}\left\|\mathbf{p}_{j}-\mathbf{y}_{i}\right\|^{2}\right]}$$

8.

$$f(\mathbf{x})=-\ln L_{\mathrm{sym}}(\mathbf{x})\sim\mathcal{GP}\left(\mu,\,k\right)$$

9.

$$k(\mathbf{x},\mathbf{x}^{\prime})=\sigma^{2}\left(1+\sqrt{5}\,r+\frac{5}{3}r^{2}\right)e^{-\sqrt{5}\,r},\quad r^{2}=\sum_{d=1}^{4}\frac{\left(x_{d}-x_{d}^{\prime}\right)^{2}}{\ell_{d}^{2}}$$

10.

$$\{\mathbf{x}_{1},\dots,\mathbf{x}_{4}\}=\arg\max\;\ln\mathbb{E}\left[\max_{j}\left(f(\mathbf{x}_{j})-f^{*}\right)_{+}\right]$$
