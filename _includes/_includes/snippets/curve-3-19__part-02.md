<div class="nb-snippet">

```python
curve_sofr.plot("1b", right=dt(2025, 3, 31))
```

```text
(<Figure size 640x480 with 1 Axes>,
 <Axes: >,
 [<matplotlib.lines.Line2D at 0x1a423dbba90>])
```

![output](/assets/snippets/curve-3-19/part-02/curve-3-19__part-02__cell033__out01.png)

```python
curve_sofr.rate(dt(2025, 3, 21), dt(2025, 3, 22)).real
```

```text
4.326195119102039
```

</div>
