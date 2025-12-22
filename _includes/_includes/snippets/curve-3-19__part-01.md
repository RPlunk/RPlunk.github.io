<div class="nb-snippet">

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

```python
order_books_by_symbol["SR3M5"]
```

<div class="nb-output-html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bid Size</th>
      <th>Bid Price</th>
      <th>Ask Price</th>
      <th>Ask Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1137.0</td>
      <td>95.890</td>
      <td>95.895</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1021.0</td>
      <td>95.885</td>
      <td>95.900</td>
      <td>1390.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1141.0</td>
      <td>95.880</td>
      <td>95.905</td>
      <td>1253.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>864.0</td>
      <td>95.875</td>
      <td>95.910</td>
      <td>1211.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1360.0</td>
      <td>95.870</td>
      <td>95.915</td>
      <td>1188.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1280.0</td>
      <td>95.865</td>
      <td>95.920</td>
      <td>1233.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1185.0</td>
      <td>95.860</td>
      <td>95.925</td>
      <td>1013.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1243.0</td>
      <td>95.855</td>
      <td>95.930</td>
      <td>1028.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1205.0</td>
      <td>95.850</td>
      <td>95.935</td>
      <td>1008.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1156.0</td>
      <td>95.845</td>
      <td>95.940</td>
      <td>1156.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>
