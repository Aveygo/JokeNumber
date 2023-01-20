# JokeNumber
Time to simplify a joke to a number with machine learning.

Using BERT encodings and dimensionality reduction, a joke can be transfered to a single number.
This has the added benefit that similar jokes have similar numbers, unlike hashing which will result
in a more random number.

To use:

```
git clone https://github.com/Aveygo/JokeNumber.git
cd JokeNumber
python3 convert.py "Why did the chicken cross the road? To get to the other side."
```

Some results:

```
(1). Why did the chicken cross the road? To get to the other side. => 2533712
(2). Why did the cow cross the road? To get to the other side. => 2761839
(3). My dad has a heart of a lion ...and a lifetime ban from the local zoo. => 2457741

| (1) - (2) | = 228127
| (2) - (3) | = 304098

```

As expected, the first two jokes are more similar than the last two

```
A man is sent to prison for the first time.

The first night there, after the lights in the cell block are turned off, 
he immediately sees his cellmate going over to the bars and yelling, "twelve!"

The whole cell block breaks out laughing. A few minutes later, somebody else in 
the cell block yells, "twenty-three!" Again, the whole cell block breaks out laughing.

"Why are you guys just yelling numbers?" He asks his cellmate. "What's so funny 
about random numbers?"

"Well," says the older prisoner, "They're not random. It's just that we've all been 
in this here sub for so long, we all know all the same jokes. So after a while we 
just started giving them numbers and yelling those numbers is enough to remind us 
of the joke instead of telling it."

 Wanting to fit in, the new prisoner walks up to the bars and yells, "SIX!" But 
 instead of laughter, a dead silence falls on the cell block. He turns to the older 
 prisoner, "What's wrong? Why didn't I get any laughs?"

"You didn't tell it right."
```
Is joke #6489523
