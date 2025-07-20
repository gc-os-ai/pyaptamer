from pyaptamer.pseaac import PSeAAC

# Example protein sequence
sequence = "ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR"

# Create a PSeAAC instance
# - Use all 21 properties
# - Group into sets of 3 (7 groups)
pseaac = PSeAAC(prop_indices=[0, 5, 6, 7])

# Generate feature vector
features = pseaac.transform(sequence)

# Display results
print(f"Input sequence: {sequence}")
print(f"Feature vector length: {len(features)}")
print(f"First 10 features:\n{features[:10]}")
