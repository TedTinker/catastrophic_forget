# Catastrophic Forgetting

Intelligent entities learn continuously, and can remember their important lessons from long ago.

Neural networks do not have this power, and can succumb to catastrophic forgetting. 

The main.py file makes a pytorch model which can learn to classify MNIST digits with over 90% accurasy. But, it first trains only on the digits zero through four, then trains only on the digits five through nine. Although it learns to identify digits zero through four, it forgets how. 
