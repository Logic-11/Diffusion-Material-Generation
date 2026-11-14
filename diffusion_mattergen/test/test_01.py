import torch

def frac_to_cart(frac, lattices):
    '''frac:(N,3), lattices:(3,3)'''
    cart = frac @ lattices
    return cart

def cart_to_frac(cart, lattices):
    frac = cart @ torch.linalg.inv(lattices)
    return frac

Z = torch.tensor([11, 17], dtype=torch.long)
L = torch.tensor([[2, 0, 0], 
                  [1, 3, 0],
                  [0, 0, 4]], dtype=torch.float32)
F = torch.tensor([[0, 0, 0],
                  [0.5, 0.5, 0.25]], dtype=torch.float32)

print(f"Z: shape {Z.shape}, dtype {Z.dtype}")
print(f"L: shape {L.shape}, dtype {L.dtype}")
print(f"F: shape {F.shape}, dtype {F.dtype}")

X = frac_to_cart(F, L)
reconstructed_F = cart_to_frac(X, L)
torch.testing.assert_close(reconstructed_F, F)
print(X)
print(reconstructed_F)
