## A SOFR IRS Pricing Curve in Rateslib 

### References 

(1) J H M Darbyshire. Pricing and Trading Interest Rate Derivatives (v3). [Amazon link](https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538/ref=pd_bxgy_d_sccl_1/147-3131794-6501123?pd_rd_w=c0uHB&content-id=amzn1.sym.dcf559c6-d374-405e-a13e-133e852d81e1&pf_rd_p=dcf559c6-d374-405e-a13e-133e852d81e1&pf_rd_r=1WPX4TY3T0K8PC6V2KKS&pd_rd_wg=lEsF0&pd_rd_r=59d0553e-6ab6-4cc0-8a0a-ab93a1e2a82b&pd_rd_i=0995455538&psc=1) 

(2) J H M Darbyshire. Coding Interest Rates: FX, Swaps & Bonds. [Amazon link](https://www.amazon.com/dp/0995455562) 

(3) [Rateslib Documentation][URL1] 

[URL1]: https://rateslib.com/py/en/2.0.x/index.html 

(4) [Code Repository for Pricing and Trading IRDs][URL2] 

[URL2]: https://github.com/attack68/book_irds3 

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

![order books by symbol](/assets/img1.PNG)

![mid price results](/assets/img2.PNG)

Each combo price implies a linear relationship between the component outright prices; however, our calculated mid-prices may not be entirely consistent. This may be due to noisy individual book mid prices, or possibly due to liquidity that may be “hidden” across orders implied through related books. Least squares regression can be used to obtain a best-fit coherent set of core outright prices potentially using greater weights for more stable, independent sources. This is shown in (1 ch.16) and implemented to obtain our set of core prices which are then converted into rate format. 

```python
def calculate_core_prices(A, s, w=None):
   
    if w is not None:
        W = np.diag(w)
        p_star = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ s)
    else:
        p_star = np.linalg.pinv(A) @ s
    return p_star

A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
              [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1]])
s = []

for _, row in results_df.iterrows():
    price = row["Mid-Price"]
    if isinstance(price, float):
        if "-" in row["Instrument"]:
            price = price * 0.01  # If instrument is a spread, convert from ticks to price
        s.append(price)

s = np.array(s)

assert np.linalg.matrix_rank(A) == A.shape[1] #Check for full column rank

core_prices = calculate_core_prices(A, s, w = [1.0] * 16 + [0.2] * 2 + [0.5] * 12 + [0.2] * 11)

print("Core Implied Prices:", core_prices)
```

![core rates](/assets/img3.PNG)

These rates are then used to build a discount factor based SOFR futures curve which will be later used to price convexity adjusted swap rates. Rateslib provides extensive documentation including a sequentially built [User Guide](https://rateslib.com/py/en/2.0.x/i_guide.html) with plenty of examples. Importantly, defining curves with given node dates as degrees of freedom and then numerically solving curves based on calibrating instruments are two separate processes. 

```python
curve_sofr = Curve(
    nodes={
        dt(2025, 3, 19): 1.0,
        dt(2025, 4, 1): 1.0,
        dt(2025, 5, 1): 1.0,
        dt(2025, 6, 1): 1.0,
        dt(2025, 6, 18): 1.0,
        dt(2025, 9, 17): 1.0,
        dt(2025, 12, 17): 1.0,
        dt(2026, 3, 18): 1.0,
        dt(2026, 6, 17): 1.0,
        dt(2026, 9, 16): 1.0,
        dt(2026, 12, 16): 1.0,
        dt(2027, 3, 17): 1.0,
        dt(2027, 6, 16): 1.0,
        dt(2027, 9, 15): 1.0,
        dt(2027, 12, 15): 1.0,
        dt(2028, 3, 15): 1.0,
        dt(2028, 6, 21): 1.0,
    },
    interpolation="log_linear",
    calendar="nyc",
    convention="act360",
    modifier="MF",
    id="sofr",
)
```

Curves can be instantiated with various attributes as shown above, leveraging built in modules. Interpolation between nodes defaults to log linear for the Curve class (specified explicitly here), where the log of discount factors between nodes is linearly interpolated producing constant overnight rates as a function of time. Spline interpolation is also available with the ability to mix interpolation methods between different tenors of the curve.

Here the curve is created with an NYC calendar (custom and combined calendars can also be used), actual/360 day count convention, a day modifier rule of modified following and a string identification of “sofr”. The initial node is set to today’s date with subsequent nodes set to futures expiration dates. 

```python
sofr_1903 = pd.DataFrame(
    data=[4.33, 4.33, 4.34, 4.35, 4.34, 4.33, 4.32, 4.31, 4.3, 4.3, 4.32, 4.31],
    index=pd.Index(["03-03-2025", "04-03-2025", "05-03-2025", "06-03-2025", "07-03-2025", "10-03-2025", "11-03-2025", "12-03-2025", "13-03-2025",
                    "14-03-2025", "17-03-2025", "18-03-2025"], name="reference_date"),
    columns=["rate"]
)
```

```python
sofr_1903.to_csv("sofr_1903.csv")
```

```python
defaults.fixings.directory = os.getcwd()
```

```python
defaults.fixings["sofr_1903"]
```

```python
args_1m = dict(spec="usd_stir1", curves="sofr")
args_3m = dict(spec="usd_stir", curves="sofr")
sofr_futures = [
    STIRFuture(dt(2025, 3, 1), dt(2025, 4, 1), leg2_fixings=defaults.fixings["sofr_1903"], **args_1m),
    STIRFuture(dt(2025, 4, 1), dt(2025, 5, 1), **args_1m),
    STIRFuture(dt(2025, 5, 1), dt(2025, 6, 1), **args_1m),
    STIRFuture(effective=get_imm(code="H25"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="M25"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="U25"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="Z25"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="H26"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="M26"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="U26"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="Z26"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="H27"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="M27"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="U27"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="Z27"), termination="3m", **args_3m),
    STIRFuture(effective=get_imm(code="H28"), termination="3m", **args_3m),
]
```
![futures conventions](/assets/img4.PNG)

A wide variety of fixed income securities and derivative [instruments](https://rateslib.com/py/en/2.0.x/g_instruments.html) can be constructed in rateslib with deep flexibility to include multi-currency instruments and foreign exchange value conversions. Advantageously, default instrument specification is available for standard market conventions and here our STIR futures have been assigned one-month and three-month SOFR futures conventions. Daily SOFR fixings for the displayed reference dates obtained from the [New York Fed webpage](https://www.newyorkfed.org/markets/reference-rates/sofr) have also been manually created and provided for the March one-month future. 

A string id mapping to the “sofr” curve id has been provided to each future. Other explicit and dynamic options are available and [pricing mechanisms](https://rateslib.com/py/en/2.0.x/x_mechanisms.html) are documented thoroughly, but this essentially provides a flexible mapping to link an instrument object to other curve and solver objects. This mode is recommended as a general best practice and allows an instrument to be added to a portfolio. Separate forecasting and discounting curves can also be assigned although we will stay within the risk-free rate framework with standard USD collateral.  

```python
s_sofr =  core_rates_df["Rate"].tolist()
```

```python
instrument_labels_sofr = ["1H5", "1J5", "1K5", "H5", "M5", "U5", "Z5", "H6", "M6", "U6", "Z6", "H7", "M7", "U7", "Z7", "H8"]
```

```python
solver_sofr = Solver(
    curves=[curve_sofr],
    instruments=sofr_futures,
    s=s_sofr,
    instrument_labels=instrument_labels_sofr,
    id="sofr",
)
```

![sofr solver](/assets/img5.PNG)

Rateslib solves curves via a numerical optimization routine, not bootstrapping. This provides for a modern method that affords many advantages, one in particular being flexibility, which will be especially apparent in solving the swaps curve. The solver will iterate through solutions until the value of each node re-prices the calibrating instruments as closely as possible. The discount factor values that were assigned to each node when the curve was defined serve as the initial guess.  

The solver uses a least squares objective function that attempts to minimize the calibrating instrument rates from the solved curve and the known instrument rates that are parameters to the curve. The objective function is shown [here](https://rateslib.com/py/en/2.0.x/c_solver.html) and more advanced aspects of the solver are documented in (2). 

The solver is provided with curves, instruments, and target rates after which the optimizer updates the curve. The algorithm used defaults to Levenberg Marquardt as a blend of gradient descent and Gauss Newton, however the latter two are also available. Here the solver successfully solves the curve after reaching function tolerance. 

![futures rates](/assets/img6.PNG)

When calculating values providing only a curve is sufficient, however risk metrics require derivative information contained within a solver. The interaction between these objects is key to pricing, however the flexibility and ease of use that this provides is a major attribute of the library. 

Importantly, the library relies on its own internal implementation of automatic differentiation for calculating derivatives without external dependencies. This is an important benefit that enhances performance and is used both for calculating risk metrics and optimization. The mechanics of how this is implemented are fairly involved and are documented extensively in (2) and (3). 

This creates a dual number datatype that is central to functionality for sensitivities. Here the solved curve contains AD information and the instrument rate is returned as a dual datatype when only the curve is provided.     

![curve sofr plot](/assets/img7.PNG)

A nice feature is that built in curve plotting attributes are available for visualization of forward rates for a given tenor. The above plot shows forward rates for one business day from our curve_sofr. 

This is a discount factor based curve from which an instrument rate, with all of its appropriate conventions, can be determined. A subtle yet important distinction that the library segregates well is the difference between a *curve* rate, implied from a curve with its conventions, and an *instrument* rate that is priced off of that curve. The curve plot shows single period simple rates with the conventions of the curve. This is explained within the [curve](https://rateslib.com/py/en/2.0.x/api/rateslib.curves.Curve.html#rateslib.curves.Curve) section of (3).   

![simple rates](/assets/img8.PNG)

![simple rates](/assets/img9.PNG)

![simple rates](/assets/img10.PNG)

![simple rates](/assets/img11.PNG)

By limiting the right axes of the chart we can inspect the curve more closely where small peaks may appear around weekends and holidays. This is the curve showing a rate applicable over non-business days under an actual/360 day count convention and nyc calendar. We can see that the curve rate calculated between the 21st (Friday) and 22nd (Saturday) is constant and that the log linearly interpolated DF’s are decreasing over all days. We can also see that the rate plotted for the 21st is the simple rate applicable between the 21st and 24th (Monday).  

```python
curve_irs = Curve(
     nodes={
        dt(2025, 3, 19): 1.00,  # today's DF
        dt(2025, 3, 31): 1.00, #turn 1
        dt(2025, 4, 1): 1.00, #turn 1
        dt(2025, 5, 8): 1.00,  # defined effective FOMC dates..
        dt(2025, 6, 20): 1.00,
        dt(2025, 6, 30): 1.00, #turn 2
        dt(2025, 7, 1): 1.00, #turn 2
        dt(2025, 7, 31): 1.00,
        dt(2025, 9, 18): 1.00,
        dt(2025, 9, 30): 1.00, #turn 3
        dt(2025, 10, 1): 1.00, #turn 3
        dt(2025, 10, 30): 1.00,
        dt(2025, 12, 11): 1.00,
        dt(2025, 12, 31): 1.00, #turn 4
        dt(2026, 1, 2): 1.00, #turn 4
        dt(2026, 1, 29): 1.00,
        dt(2026, 3, 19): 1.00,
        dt(2026, 3, 31): 1.00, #turn 5
        dt(2026, 4, 1): 1.00, #turn 5
        dt(2026, 4, 30): 1.00,  
        dt(2026, 6, 18): 1.00,
        dt(2026, 6, 30): 1.00, #turn 6
        dt(2026, 7, 1): 1.00, #turn 6
        dt(2026, 7, 30): 1.00,
        dt(2026, 9, 17): 1.00,
        dt(2026, 9, 30): 1.00, #turn 7
        dt(2026, 10, 1): 1.00, #turn 7
        dt(2026, 10, 29): 1.00,
        dt(2026, 12, 10): 1.00,
        dt(2026, 12, 31): 1.00, #turn 8
        dt(2027, 1, 4): 1.00, #turn 8
        dt(2027, 1, 28): 1.00,
        dt(2027, 3, 18): 1.00, # estimated effective FOMC dates...
        dt(2027, 3, 31): 1.00, #turn 9
        dt(2027, 4, 1): 1.00, #turn 9
        dt(2027, 5, 6): 1.00,  
        dt(2027, 6, 17): 1.00,
        dt(2027, 6, 30): 1.00, #turn 10
        dt(2027, 7, 1): 1.00, #turn 10
        dt(2027, 7, 29): 1.00,
        dt(2027, 9, 16): 1.00,
        dt(2027, 9, 30): 1.00, #turn 11
        dt(2027, 10, 1): 1.00, #turn 11
        dt(2027, 11, 5): 1.00,
        dt(2027, 12, 16): 1.00,
        dt(2027, 12, 31): 1.00, #turn 12
        dt(2028, 1, 3): 1.00, #turn 12
        dt(2028, 1, 27): 1.00,
        dt(2028, 3, 9): 1.00,
        dt(2028, 3, 31): 1.00, #turn 13
        dt(2028, 4, 3): 1.00, #turn 13
        dt(2028, 4, 20): 1.00,
        dt(2028, 6, 21): 1.00,  # final IMM.
    },
    interpolation="log_linear",
    calendar="nyc",
    convention="act360",
    modifier="MF",
    id="irs",
)
```

We can now build a more granular swaps curve, calibrated by market data, that is intended for the accurate pricing of cashflows. This curve will terminate slightly after the 3-year tenor both due to the availability of market data and to increase focus on the short end. Under the premise that central bank rate expectations are a primary driver of short end rates we will choose to maintain log-linear interpolation with node dates set as policy effective dates. This models the evolution of overnight forward rates as a step function with constant overnight rates between nodes and jumps on FOMC effective dates. Nodes marked as defined dates have been taken from the [fed website](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) meeting calendar. Nodes beyond the furthest defined meeting (January 28th, 2027 at time of construction) have been estimated following the eight per year, six-week interval, one month off per quarter convention typically followed by the FOMC. The *effective* date has been chosen as the next business day after the typical Wednesday meeting announcement.  

We will also implement quarter and year end turns by adding two nodes for each as shown in (2). These “turns” represent specific dates where rates are expected to increase based on some calendar driven constraint, which is usually explained based on regulatory purposes. A month end turn, as has sometimes appeared in the SOFR overnight market, could also be implemented in a similar manner although we will only include quarter/year end here. Turns are a nuanced aspect of the curve which are difficult to represent based on market pricing and each turn will be manually set as a one day 5 basis point jump. These will be set using spread instruments for dates immediately preceding and following the turn. This method can only be used under log-linear interpolation, however methods of implementation for log-cubic interpolation and fading turns over time are shown in (2).  

```python
args_turn = dict(termination="1d", frequency="A", curves="irs")
args_fomc = dict(spec="usd_irs", curves="irs")
fomc_1 = IRS(dt(2025, 5, 8), dt(2025, 6, 20), **args_fomc)
fomc_2 = IRS(dt(2025, 6, 20), dt(2025, 7, 31), **args_fomc)
fomc_3 = IRS(dt(2025, 7, 31), dt(2025, 9, 18), **args_fomc)
fomc_4 = IRS(dt(2025, 9, 18), dt(2025, 10, 30), **args_fomc)
fomc_5 = IRS(dt(2025, 10, 30), dt(2025, 12, 11), **args_fomc)
fomc_6 = IRS(dt(2025, 12, 11), dt(2026, 1, 29), **args_fomc)
fomc_7 = IRS(dt(2026, 1, 29), dt(2026, 3, 19), **args_fomc)
fomc_8 = IRS(dt(2026, 3, 19), dt(2026, 4, 30), **args_fomc)
fomc_9 = IRS(dt(2026, 4, 30), dt(2026, 6, 18), **args_fomc)
fomc_10 = IRS(dt(2026, 6, 18), dt(2026, 7, 30), **args_fomc)
fomc_11 = IRS(dt(2026, 7, 30), dt(2026, 9, 17), **args_fomc)
fomc_12 = IRS(dt(2026, 9, 17), dt(2026, 10, 29), **args_fomc)
fomc_13 = IRS(dt(2026, 10, 29), dt(2026, 12, 10), **args_fomc)
fomc_14 = IRS(dt(2026, 12, 10), dt(2027, 1, 28), **args_fomc)
fomc_15 = IRS(dt(2027, 1, 28), dt(2027, 3, 18), **args_fomc)
fomc_16 = IRS(dt(2027, 3, 18), dt(2027, 5, 6), **args_fomc)
fomc_17 = IRS(dt(2027, 5, 6), dt(2027, 6, 17), **args_fomc)
fomc_18 = IRS(dt(2027, 6, 17), dt(2027, 7, 29), **args_fomc)
fomc_19 = IRS(dt(2027, 7, 29), dt(2027, 9, 16), **args_fomc)
fomc_20 = IRS(dt(2027, 9, 16), dt(2027, 11, 5), **args_fomc)
fomc_21 = IRS(dt(2027, 11, 5), dt(2027, 12, 16), **args_fomc)
fomc_22 = IRS(dt(2027, 12, 16), dt(2028, 1, 27), **args_fomc)
fomc_23 = IRS(dt(2028, 1, 27), dt(2028, 3, 9), **args_fomc)
fomc_24 = IRS(dt(2028, 3, 9), dt(2028, 4, 20), **args_fomc)
fomc_25 = IRS(dt(2028, 4, 20), dt(2028, 6, 21), **args_fomc)
turn_1a = IRS(effective=dt(2025, 3, 30), **args_turn)
turn_1b = IRS(effective=dt(2025, 3, 31), **args_turn)
turn_1c = IRS(effective=dt(2025, 4, 1), **args_turn)
turn_2a = IRS(effective=dt(2025, 6, 29), **args_turn)
turn_2b = IRS(effective=dt(2025, 6, 30), **args_turn)
turn_2c = IRS(effective=dt(2025, 7, 1), **args_turn)
turn_3a = IRS(effective=dt(2025, 9, 29), **args_turn)
turn_3b = IRS(effective=dt(2025, 9, 30), **args_turn)
turn_3c = IRS(effective=dt(2025, 10, 1), **args_turn)
turn_4a = IRS(effective=dt(2025, 12, 30), **args_turn)
turn_4b = IRS(effective=dt(2025, 12, 31), **args_turn)
turn_4c = IRS(effective=dt(2026, 1, 2), **args_turn)
turn_5a = IRS(effective=dt(2026, 3, 30), **args_turn)
turn_5b = IRS(effective=dt(2026, 3, 31), **args_turn)
turn_5c = IRS(effective=dt(2026, 4, 1), **args_turn)
turn_6a = IRS(effective=dt(2026, 6, 29), **args_turn)
turn_6b = IRS(effective=dt(2026, 6, 30), **args_turn)
turn_6c = IRS(effective=dt(2026, 7, 1), **args_turn)
turn_7a = IRS(effective=dt(2026, 9, 29), **args_turn)
turn_7b = IRS(effective=dt(2026, 9, 30), **args_turn)
turn_7c = IRS(effective=dt(2026, 10, 1), **args_turn)
turn_8a = IRS(effective=dt(2026, 12, 30), **args_turn)
turn_8b = IRS(effective=dt(2026, 12, 31), **args_turn)
turn_8c = IRS(effective=dt(2027, 1, 4), **args_turn)
turn_9a = IRS(effective=dt(2027, 3, 30), **args_turn)
turn_9b = IRS(effective=dt(2027, 3, 31), **args_turn)
turn_9c = IRS(effective=dt(2027, 4, 1), **args_turn)
turn_10a = IRS(effective=dt(2027, 6, 29), **args_turn)
turn_10b = IRS(effective=dt(2027, 6, 30), **args_turn)
turn_10c = IRS(effective=dt(2027, 7, 1), **args_turn)
turn_11a = IRS(effective=dt(2027, 9, 29), **args_turn)
turn_11b = IRS(effective=dt(2027, 9, 30), **args_turn)
turn_11c = IRS(effective=dt(2027, 10, 1), **args_turn)
turn_12a = IRS(effective=dt(2027, 12, 30), **args_turn)
turn_12b = IRS(effective=dt(2027, 12, 31), **args_turn)
turn_12c = IRS(effective=dt(2028, 1, 3), **args_turn)
turn_13a = IRS(effective=dt(2028, 3, 30), **args_turn)
turn_13b = IRS(effective=dt(2028, 3, 31), **args_turn)
turn_13c = IRS(effective=dt(2028, 4, 3), **args_turn)
```

The standard problem this presents is that the curve becomes an underspecified optimization problem, as we now have more nodes (degrees of freedom) than instruments to calibrate those nodes. We can expect the solver to reach one of many possible unconstrained solutions with a curve that oscillates excessively. Different options exist to regularize this output and these can be complex and sometimes difficult to implement with advantages and disadvantages for each. Fortunately, minimizing curvature by introducing butterfly spreads targeted to price at zero is readily implemented in rateslib where an example is shown in (2). Basically, butterflies across policy periods, which can be seen to proxy a second derivative as a spread of spreads, will be targeted to price as close to zero as possible while remaining consistent with other market instruments used as actual inputs to the curve. A low weighting to the butterfly instruments can be easily added to the solver with the intention being that these instruments will be free to adjust in order to re-price market inputs more closely. This will also allow risk to be viewed against the regularizing instruments in a manner which can be interpreted. 

This weighting becomes a subjective tradeoff between smoothness and fit to market instruments. Setting the butterflies to be adjacent to each other, rather than overlapping, also seems to re-price instruments more closely while creating more curvature at the connecting points.   

```python
args_irs = dict(spec="usd_irs", curves="irs")
instruments_irs = [
    IRS(dt(2025, 3, 1), dt(2025, 4, 1), leg2_fixings=defaults.fixings["sofr_1903"], **args_irs),
    IRS(dt(2025, 4, 1), dt(2025, 5, 1), **args_irs),
    IRS(dt(2025, 5, 1), dt(2025, 6, 1), **args_irs),
    IRS(effective=get_imm(code="H25"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="M25"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="U25"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="Z25"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="H26"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="M26"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="U26"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="Z26"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="H27"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="M27"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="U27"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="Z27"), termination="3m", roll="imm", **args_irs),
    IRS(effective=get_imm(code="H28"), termination="3m", roll="imm", **args_irs),
    Spread(turn_1a, turn_1b),
    Spread(turn_1b, turn_1c),
    Spread(turn_2a, turn_2b),
    Spread(turn_2b, turn_2c), 
    Spread(turn_3a, turn_3b),
    Spread(turn_3b, turn_3c),
    Spread(turn_4a, turn_4b),
    Spread(turn_4b, turn_4c),
    Spread(turn_5a, turn_5b),
    Spread(turn_5b, turn_5c),
    Spread(turn_6a, turn_6b),
    Spread(turn_6b, turn_6c),
    Spread(turn_7a, turn_7b),
    Spread(turn_7b, turn_7c), 
    Spread(turn_8a, turn_8b), 
    Spread(turn_8b, turn_8c),
    Spread(turn_9a, turn_9b),
    Spread(turn_9b, turn_9c),
    Spread(turn_10a, turn_10b), 
    Spread(turn_10b, turn_10c), 
    Spread(turn_11a, turn_11b),
    Spread(turn_11b, turn_11c),
    Spread(turn_12a, turn_12b),
    Spread(turn_12b, turn_12c), 
    Spread(turn_13a, turn_13b), 
    Spread(turn_13b, turn_13c),
    Fly(fomc_1, fomc_2, fomc_3),
    Fly(fomc_3, fomc_4, fomc_5),
    Fly(fomc_5, fomc_6, fomc_7),
    Fly(fomc_7, fomc_8, fomc_9),
    Fly(fomc_9, fomc_10, fomc_11),
    Fly(fomc_11, fomc_12, fomc_13),
    Fly(fomc_13, fomc_14, fomc_15),
    Fly(fomc_15, fomc_16, fomc_17),
    Fly(fomc_17, fomc_18, fomc_19),
    Fly(fomc_19, fomc_20, fomc_21),
    Fly(fomc_21, fomc_22, fomc_23),
    Fly(fomc_23, fomc_24, fomc_25),
]
```

```python
convx_adj = [0, -0.00003, -0.00017, -0.00001, -0.00098, -0.00258, -0.00481, -0.00764, -0.01107, -0.01508, -0.01966, -0.02479, -0.03046, -0.03666, 
             -0.04337, -0.05116]
s_adj = [
 inst.rate(solver=solver_sofr, curves="sofr").real + convx_adj for (inst,convx_adj) in zip(instruments_irs[:16], convx_adj)
]
print(s_adj)
```

Our market input instruments will be swaps mirroring SOFR futures dates starting with three one-month instruments followed by thirteen IMM swaps. There is a good discussion in (1) on the choices involved in the number of IMM instruments that are used. In relation to the SOFR market, the three one-month instruments have been chosen to specify the very front of the curve with electronic real time instruments instead of manually input overnight or term rates. This takes advantage of pricing from one-month futures where volume and open interest seem to remain the highest in the front three contracts. Maintaining these front one-month contracts also allows us to roll out of the first three-month future once it has progressed through the first month of its reference period. After which volume in that contract tends to drop. This aims to utilize liquid exchange traded instruments while maintaining a length that extends past the 3-year tenor to a maximum of 5 months. Par tenor instruments could then be used further out on the curve. 

These instruments are then priced off the SOFR futures curve with a convexity adjustment as shown. These adjustments have been calculated with a one factor Hull-White model using a mean reversion factor of .03 and a sigma value equivalent to 101.4 bps. An example of this calculation is shown in `Convx Adj hw`. This creates a theoretical no arbitrage adjustment between the linear risk profile futures contracts and the non-linear risk profile of the swaps based on the value of convexity (gamma) in those swaps. This value would normally be determined by calibrating the model to SOFR volatility instruments, such as SOFR futures options, where a higher level of implied volatility would create more convexity value in the swaps. This is a key aspect of this pricing framework that rests on model assumptions and could certainly be improved upon here by potentially using a two-factor model and calibrating to market instruments. Actual pricing differences between futures and swaps may include other considerations, such as “cross valuation adjustments”, but that is not approached here.  

```python
s_irs = s_adj + [5, -5] * 13 + [0] * 12
```

```python
weights_irs = [1] * 16 + [1e-09] * 2 + [1] * 24 + [1e-09] * 12 #Includes low weighting for first turn to favor mkt instruments 
```

```python
instrument_labels_irs = [
    "1H5", "1J5", "1K5", "H5", "M5", "U5", "Z5", "H6", "M6", "U6", "Z6", "H7", "M7", "U7", "Z7", "H8", "turn1_left", "turn1_right", 
    "turn2_left", "turn2_right", "turn3_left", "turn3_right", "turn4_left", "turn4_right", "turn5_left", "turn5_right", "turn6_left", 
    "turn6_right", "turn7_left", "turn7_right", "turn8_left", "turn8_right", "turn9_left", "turn9_right", "turn10_left", "turn10_right",
    "turn11_left", "turn11_right", "turn12_left", "turn12_right", "turn13_left", "turn13_right", "cv1", "cv2", "cv3", "cv4", "cv5", "cv6", 
    "cv7", "cv8", "cv9", "cv10", "cv11", "cv12",
]
```

```python
solver_irs = Solver(
    pre_solvers=[solver_sofr],
    curves=[curve_irs],
    instruments=instruments_irs,
    s=s_irs,
    weights=weights_irs,
    instrument_labels=instrument_labels_irs,
    func_tol=1e-08,  
    conv_tol=1e-10,
    id="irs",
)
```

These adjusted rates will be used in our solver along with turn spread rates and butterfly rates (in basis points). Weights are also created, and here the first turn in the curve has also been given a low weighting. This first turn appears at the very front of the curve and has a large impact on the first one-month instrument expiring at the end of March. This ideally allows the turn more freedom to adjust from the set 5 basis points, and market instruments to re-price more closely. 

The solver for the swaps curve has also been given a pre solver which creates a linking chain between them. Examples are shown in (3) of how this is especially useful for creating a dependency chain between curves in different currencies. Here, this will allow us to price futures and swaps together in a single solver by assigning the correct string id curve to those respective instruments. The function tolerance and convergence tolerance of the solver have been widened to less restrictive values from their default values of 1e-12 and 1e-17 to allow the now overspecified curve to solve; and the solver reaches convergence on a function value of approx. 1.2e-6.   

![irs solver](/assets/img12.PNG)

![irs solver](/assets/img13.PNG)

![irs solver](/assets/img14.PNG)

![curve irs plot](/assets/img15.PNG)

Comparing plots of overnight forward rates of the futures curve and swaps curve shows the characteristically higher futures curve relative to swaps. 

![rates and npv](/assets/img16.PNG)

The rates for both the June IMM IRS and 2-year IRS are the mid-market swap rates for those instruments with the assigned string id “irs” curve. The June futures contract (with assigned “sofr” curve) can also be priced off the futures curve by the same solver through the pre solver. This will become especially useful when aggregating these instruments into a portfolio. The mid-market 2-year IRS is also shown with a net present value of zero as the sum of fixed and float leg npv’s. 

![cashflows and spec](/assets/img17.PNG)

![cashflows and spec](/assets/img19.PNG)

A cashflows method is available to quickly and concisely display the properties of each period for each leg of a swap, and a cashflows table is available to sum cashflows for each leg based on payment date. Without a fixed rate or notional specified, the cashflows for the 2-year are displayed for a mid-market payer with a notional of 1 million (setting a negative notional would switch to a receiver). These swaps use the default usd_irs spec but could easily be customized in a wide variety of means which are thoroughly documented in (3). 

![delta](/assets/img18.PNG)

Prioritizing risk sensitivities is a stated [pillar](https://rateslib.com/py/en/2.0.x/i_about.html#five-pillars-of-rateslib-s-design-philosophy) in the design philosophy of rateslib. Implementing automatic differentiation deep within the library confers advantages for modern risk management techniques with impressive performance. This provides for a great deal of functionality in an easy to use and robust manner. 

The analytic delta method measures the sensitivity of present value to a one basis point change in the *fixed* rate, as calculated from the mathematical formula for the PV of the swap. This can be referenced as a single number for risk providing for a practical, and sometimes useful, means of measurement for total risk. It does, however, become less reliable as the fixed rate deviates from mid-market. The convention used for risk shows positive risk for the 2-year payer as a preference for rates to increase, and represents a gain per one basis point increase in rates.

The [delta](https://rateslib.com/py/en/2.0.x/j_delta.html) method provides a means of viewing risk to the individual scenarios of market rates changing for each calibrating instrument of the solver. This calculation has traditionally been done using numerical techniques, however rateslib leverages automatic differentiation to accomplish this. Accessing this type of advanced calculation in python with the speed and flexibility of AD is a significantly beneficial aspect of rateslib.    

---

The curve created above is intended to act as a *pricing* curve, calibrated to market data, in order to accurately price SOFR IRS instruments (and futures through the pre-solver). Further on in the notebook, these prices are then provided to instruments which are included in risk models; intended to measure risk in a fashion from which liquid hedges can be enacted, and PnL can be explained in a consistent fashion. 

The `Curve 3_20` notebook then evaluates the PnL of a hedged portfolio over the subsequent day. Future posts can potentially look into some of those topics. 
