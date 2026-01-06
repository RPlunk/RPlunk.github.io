## A SOFR IRS Pricing Curve in Rateslib 

### References 

(1) J H M Darbyshire. Pricing and Trading Interest Rate Derivatives (v3). [Amazon link](https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538/ref=pd_bxgy_d_sccl_1/147-3131794-6501123?pd_rd_w=c0uHB&content-id=amzn1.sym.dcf559c6-d374-405e-a13e-133e852d81e1&pf_rd_p=dcf559c6-d374-405e-a13e-133e852d81e1&pf_rd_r=1WPX4TY3T0K8PC6V2KKS&pd_rd_wg=lEsF0&pd_rd_r=59d0553e-6ab6-4cc0-8a0a-ab93a1e2a82b&pd_rd_i=0995455538&psc=1) 

(2) J H M Darbyshire. Coding Interest Rates: FX, Swaps & Bonds. [Amazon link](https://www.amazon.com/dp/0995455562) 

(3) [Rateslib Documentation][URL] 

[URL]: https://rateslib.com/py/en/2.0.x/index.html 

(4) [Code Repository for Pricing and Trading IRDs][URL] 

[URL]: https://github.com/attack68/book_irds3 

---

This post is intended to provide for a written explanation of some of the concepts and functionality exhibited in the creation of the pricing curve in the SORF-IRS-Curve repo. It will hopefully provide for a more structured understanding than which can be conveyed by code comments or markdown cells. The goal of that project was to build a short end SOFR Interest Rate Swaps pricing curve, along with various risk models from which a portfolio of swaps can be evaluated against, hedge the portfolio, and then evaluate the performance of our hedge over the course of a single day. 

Our curve will use actual historical CME SOFR Futures data as its basic building blocks courtesy of the [Databento API](https://databento.com/), which is a great resource for efficient and cost-effective access to order book level futures data as well as historical settlement prices. The focus will be on the short end (primarily out to the 3-year tenor) both due to the public availability of swaps data, as well as to highlight the advantages provided by the rateslib library in handling difficulties encountered at the short end.

In my opinion, the rateslib library, along with its supporting documentation and supplemental resources, is an excellent tool for those wanting to learn the mechanics of modern risk-free rate derivatives markets; which is ultimately the personal motivation for this project. The above references should be considered as a source for this entire work, which aims to integrate mainly risk functions from the original sandbox testing environment in (4) guided by concepts in (1). More technical documentation of architecture and algorithms for rateslib is provided by (2). 

The material presented should by no means be considered correct, and readers should refer to the mentioned sources for further explanation. My aim in sharing this work is to assist others interested in the same material, who may then be able to improve upon it. If nothing more, errors that do exist can provide feedback as to where misunderstandings may lie. 

---

The notebook Curve 3_19 builds what is essentially a closing curve for that day, and we will begin by calculating mid-market prices from a snapshot of CME SOFR futures order book data (including various spread and butterfly order books) taken at 1500 central standard time. This snapshot builds order books across the top ten price levels from the mbp-10 (market by price) schema using the CME Globex MDP 3.0 dataset. An example of how this data can be retrieved from the Databento API and then stored in a .json format is included in `Data/Multiple Books`. 

CME SOFR futures permit relatively wide tick increments of .5bps so we will define a function for a mid-market algorithm that attempts to reduce variance of mid-price due to trading taking place, or volume being added, at a particular point. The mean intrinsic depth average will take into account information beyond the first depth by averaging the bid and ask prices that would be achieved up to some specified maximum quantity. 

```python
@jit(nopython=True)
def single_sided_mida(b, w, t):
    sum_w = w.cumsum()
    t = sum_w[-1] if t > sum_w[-1] else t
    n = len(np.where(t > sum_w)[0])
    if n == 0:
        return b[0]
    else:
        sum_bw = (b * w).cumsum()
        p = b[0] * sum_w[0]
        for j in range(1, n):
            p += b[j] * (sum_w[j] - sum_w[j - 1])
            p += (sum_bw[j - 1] - b[j] * sum_w[j - 1]) * (log(sum_w[j]) - log(sum_w[j - 1]))
        p += b[n] * (t - sum_w[n - 1])
        p += (sum_bw[n - 1] - b[n] * sum_w[n - 1]) * (log(t) - log(sum_w[n - 1]))
        return p / t

def mean_intrinsic_depth_average(b, w, a, v, t):
    p_mbida = single_sided_mida(b, w, t)
    p_maida = single_sided_mida(a, v, t)
    return (p_mbida + p_maida) / 2
```

(4) Includes a function for this algorithm which also makes use of the Numba @jit decorator as a just-in-time compiler for improved performance. We then iterate over each of these order books using a maximum parameter, z, chosen here based on CME regular trading hours block trade minimum sizes. 

```python
with open("Data/order_books/all_order_books_0319.json") as f:
    books = json.load(f)

results = []
order_books_by_symbol = {}

z1_indices = set(range(16, 18))
z2_indices = set(range(0, 3)) | set(range(30, 41))
z3_indices = set(range(18, 30))
z4_indices = set(range(3, 16))  
 
#Spread and butterfly mins applied as sum of legs 
z1 = 250 #1 month spread 
z2 = 500 #1 month/3 month butterfly 
z3 = 1000 #3 month spread
z4 = 2000 #3 month 

z_by_index = {}
for i in z1_indices: z_by_index[i] = z1
for i in z2_indices: z_by_index[i] = z2
for i in z3_indices: z_by_index[i] = z3
for i in z4_indices: z_by_index[i] = z4

for i, book in enumerate(books):
    
    z = z_by_index.get(i)

    b = np.array(book["bid_prices"], dtype=float)
    w = np.array(book["bid_sizes"], dtype=float)
    a = np.array(book["ask_prices"], dtype=float)
    v = np.array(book["ask_sizes"], dtype=float)

    # Replace sentinel prices and zero sizes with NaN
    invalid_bids = (b > 1e9) | (w <= 0)
    invalid_asks = (a > 1e9) | (v <= 0)
    b[invalid_bids] = np.nan
    w[invalid_bids] = np.nan
    a[invalid_asks] = np.nan
    v[invalid_asks] = np.nan

    # Filter valid entries
    valid_bids = ~np.isnan(b) & ~np.isnan(w)
    valid_asks = ~np.isnan(a) & ~np.isnan(v)

    b_clean = b[valid_bids]
    w_clean = w[valid_bids]
    a_clean = a[valid_asks]
    v_clean = v[valid_asks]

    if len(b_clean) > 0 and len(a_clean) > 0:
        mid_price = mean_intrinsic_depth_average(b_clean, w_clean, a_clean, v_clean, z)
    else:
        mid_price = None

    results.append({
        "Instrument": book["symbol"],
        "Mid-Price": mid_price
    })

    order_books_by_symbol[book["symbol"]] = pd.DataFrame({
        "Bid Size": w,
        "Bid Price": b,
        "Ask Price": a,
        "Ask Size": v
    })

results_df = pd.DataFrame(results)
```

#### Sample Image

![My plot](/assets/imgtest1.PNG)

---

#### Another Sample Image

![My plot](/assets/imgtest2.PNG)
