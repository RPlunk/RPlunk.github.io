## Blog Post Title From First Header

Due to a plugin called `jekyll-titles-from-headings` which is supported by GitHub Pages by default. The above header (in the markdown file) will be automatically used as the pages title.

If the file does not start with a header, then the post title will be derived from the filename.

This is a sample blog post. You can talk about all sorts of fun things here.

---

### This is a header

#### Sample Code

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

#### More Sample Code

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

#### Sample Image

![My plot](/assets/imgtest1.png)

---

#### Another Sample Image

![My plot](/assets/imgtest2.png)

