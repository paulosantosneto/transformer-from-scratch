# Foundations


# Positional Encoding

Positional encoding is a fundamental tool for representing positions in a continuous sense with **good mathematical properties**. The idea is to encode each position with a unique representation in continuous space with $\mathbb{R^D}$ dimensions. This unique representation is added to the input embeddings (subword embeddings, for exemple). Therefore, it is **not necessary to deal with recurrence or convolutions in the network**.

$$
\begin{equation}
PE(pos, 2i) = \sin({\frac{pos}{10000^{\frac{2i}{d_{model}}}}})
\end{equation}
$$

$$
\begin{equation}
PE(pos, 2i + 1) = \cos({\frac{pos}{10000^{\frac{2i + 1}{d_{model}}}}})
\end{equation}
$$


It's important to note that positional encoding is related to the **length of the input** and is usually predefined with a threshold parameter. In this sense, very large entries will have inputs cut off or will not be accepted.


Mathematicaly, positional encoding is a type of **interpolation of manifolds task** [1]. And in the end, this is done with a pair of sin and cos that map the binarization into continuos functions, and **allow linear transformations operations** between positions.

$$
\begin{equation}
\vec{p_t}^{(i)} = \begin{pmatrix}
 sin(w_1 \cdot t) \\ cos(w_1 \cdot t) \\ \vdots \\ sin(w_{\frac{D}{2}}\cdot t) \\ cos(w_{\frac{D}{2}} \cdot t)
\end{pmatrix}, \text{where } w_k = \frac{1}{10000^{\frac{2k}{D}}}  
\end{equation}
$$

For more information on the derivation of the above formula (3), it is useful to read these posts [2][3].

When moving to code, there is a small manipulation in order to simplify the calculation. Basically, the **logarithmic and exponent properties are applied to eliminate the exponents in $w_k$**. Below, you can see the code needed to build your own positional encoding.


```python
# defining positions (pos) and frequencies (wk)
pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
wk = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

# empty matrix that will be filled with the values of sines and cosines.
self.pos_enc = torch.zeros((max_len, d_model))

# filling in sines for even positions and cosines for odd positions.
self.pos_enc[:, 0::2] = torch.sin(pos * wk)
self.pos_enc[:, 1::2] = torch.cos(pos * wk)
```

At the end, the positional encoding can be added or concatenated to the word embeddings. There is a lot of speculation about which one is more powerful and the mathematical aspects behind it [4], but both work well, although added with word embeddings are more common.


![Positinal Encoding Heatmap.](https://github.com/paulosantosneto/transformer-variants/blob/main/notes/figures/pos_enc_heatmap.png)



## References

[1] [Master Positional Encoding, by Jonathan Kernes.](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)

[2] [Transformer Architecture: The Positional Encoding, by Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

[3] [Linear Relationships in the Transformer’s Positional Encoding, by Timo Denk](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/)

[4] [Position Encoding in Transformer [Reddit]](https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/)

