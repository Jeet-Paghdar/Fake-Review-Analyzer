"""
generate_data.py
Generates a realistic synthetic dataset of Authentic and Fake/Deceptive reviews.
Run this first before train_model.py
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Fake Review Vocabulary ─────────────────────────────────────────────────────

FAKE_OPENERS = [
    "This product is absolutely amazing!",
    "Best purchase I have ever made!!!",
    "I LOVE this product so much!",
    "Incredible product, highly recommend to everyone!",
    "Perfect in every single way, no complaints!",
    "This is the BEST thing I have ever bought in my life!",
    "Outstanding quality and outstanding value!",
    "Five stars is simply not enough!!!",
    "Wow wow wow, what an incredible product!",
    "Totally exceeded all of my expectations!!",
    "Fantastic, fantastic, fantastic!",
    "I cannot believe how great this truly is!",
    "This has completely changed my life!",
    "Super super happy with this amazing purchase!",
    "Greatest product on the entire market without a doubt!",
    "Absolutely perfect, zero issues whatsoever!",
    "I am blown away by the quality of this item!",
    "This is hands down the best product I have reviewed!",
]

FAKE_MIDDLES = [
    "It works exactly as described and even better than expected.",
    "The quality is top notch and the material feels incredibly premium.",
    "Shipping was so fast and the packaging was beautifully done.",
    "I have already recommended this to all my friends and family members.",
    "I will definitely be buying more of these products again very soon.",
    "This is worth every single penny, do not hesitate to buy.",
    "Customer service was also really helpful, friendly and professional.",
    "I have tried many similar products but none can compare to this one.",
    "The instructions were perfectly clear and setup took only two minutes.",
    "I am so impressed I left a five star review for the very first time ever.",
    "The design is sleek, modern and looks absolutely beautiful in my home.",
    "I was skeptical but now I am a true believer in this amazing brand.",
    "Everyone who sees it at my house asks me where I bought it.",
    "Works flawlessly every single time without any issues at all.",
    "I have used it every day since it arrived and it never disappoints.",
]

FAKE_CLOSERS = [
    "Buy it now, you will not regret it one bit!",
    "10 out of 10, absolutely perfect in every way!",
    "A must buy for absolutely everyone!",
    "100% recommend to anybody and everybody!",
    "Do yourself a favor and buy this today without hesitation!",
    "Just buy it, seriously, you will thank me later!",
    "You will thank yourself for making this decision!",
    "Greatest purchase of my entire life, no exaggeration!",
    "Stop reading reviews and just add it to your cart right now!",
    "Five stars all day every day, forever!!!",
    "Trust me, this is the only product you will ever need!",
    "Do not waste another second, just get it now!!!",
]

# ── Authentic Review Vocabulary ────────────────────────────────────────────────

AUTHENTIC_OPENERS = [
    "I have been using this for about three weeks now.",
    "Bought this after reading several mixed reviews online.",
    "This was a gift for my partner who needed it for work.",
    "Ordered this to replace my old one that stopped working.",
    "Been looking for something like this for a while now.",
    "I was skeptical at first given the relatively high price point.",
    "My colleague has a similar model so I decided to try this brand.",
    "Got this during the sale and finally got around to using it.",
    "I have gone through two of these over the past year and a half.",
    "Decided to try this after a friend recommended the brand.",
    "Needed something specific for my use case so chose this model.",
    "Replaced my previous one with this after it broke after two years.",
    "Did a lot of research before settling on this particular model.",
    "Picked this up on a whim and have been pleasantly surprised.",
    "Using this for my small home office setup and it fits well.",
]

AUTHENTIC_MIDDLES = [
    "The build quality is decent but not exceptional. The plastic parts feel a bit hollow.",
    "Works as advertised for the most part. Had a minor issue setting it up on the first day.",
    "The size is slightly smaller than the photos suggest, so measure carefully before buying.",
    "For the price, I think it offers fair value. Just do not expect premium materials.",
    "Noticed a faint smell when first unpacking it but it faded after a few days of use.",
    "The instructions could be a lot clearer but there are good tutorials available online.",
    "Battery life is about eighty percent of what they claim, which is still acceptable to me.",
    "Works well on smooth surfaces but struggles a bit on thick carpet, keep that in mind.",
    "The buttons feel a little stiff initially but I am told they loosen up over time with use.",
    "Customer support was slow to respond initially but eventually resolved my issue satisfactorily.",
    "Has a slight learning curve but once you understand it, it becomes quite useful.",
    "Three of my colleagues have the same product and we have had mixed experiences with it.",
    "The color in person looks slightly different from what is shown in the product photos.",
    "Arrived with minor scratches on the bottom but nothing visible during normal use.",
    "Setup took about thirty minutes which is longer than I expected for a product like this.",
    "The app that pairs with it is a bit clunky and could use a serious design update.",
    "Works perfectly fine for casual use but power users might find it a bit limited.",
    "Noticed it gets slightly warm after extended use, which is worth being aware of.",
]

AUTHENTIC_CLOSERS = [
    "Overall I would recommend it with the caveats mentioned above.",
    "Happy with the purchase overall, though it is definitely not perfect.",
    "Good value for money if your expectations going in are realistic.",
    "Would buy again, but hoping they improve the design in future versions.",
    "Solid choice for this price range. Not life changing, but gets the job done.",
    "I would give it 3.5 stars if I could. Settled on 4 for now.",
    "Time will tell if it holds up well, but so far things are looking good.",
    "Probably not for everyone, but it suits my specific needs quite well.",
    "Cautiously recommended. Good product but do read the other reviews too.",
    "Decent product with some room for improvement. Would still choose it again.",
]

PRODUCTS = [
    "wireless earbuds", "laptop stand", "mechanical keyboard", "phone case",
    "portable charger", "desk lamp", "webcam", "bluetooth speaker",
    "water bottle", "running shoes", "yoga mat", "coffee maker",
    "air fryer", "sleep mask", "resistance bands", "monitor arm",
    "USB hub", "ring light", "mouse pad", "cable organizer",
]


def make_fake_review():
    product = random.choice(PRODUCTS)
    opener = random.choice(FAKE_OPENERS)
    middle = " ".join(random.sample(FAKE_MIDDLES, k=random.randint(2, 4)))
    closer = random.choice(FAKE_CLOSERS)
    if random.random() > 0.4:
        middle = f"The {product} is just what I needed. " + middle
    return f"{opener} {middle} {closer}"


def make_authentic_review():
    product = random.choice(PRODUCTS)
    opener = random.choice(AUTHENTIC_OPENERS)
    middle = " ".join(random.sample(AUTHENTIC_MIDDLES, k=random.randint(3, 5)))
    closer = random.choice(AUTHENTIC_CLOSERS)
    if random.random() > 0.4:
        opener = f"The {product} arrived on time and well packaged. " + opener
    return f"{opener} {middle} {closer}"


def generate_dataset(n_fake=1200, n_authentic=1200):
    fake_reviews    = [make_fake_review()     for _ in range(n_fake)]
    authentic_reviews = [make_authentic_review() for _ in range(n_authentic)]

    df = pd.DataFrame({
        "review": fake_reviews + authentic_reviews,
        "label":  [1] * n_fake + [0] * n_authentic   # 1=Fake, 0=Authentic
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("reviews_dataset.csv", index=False)
    print(f"✅ Dataset saved: {len(df)} total rows")
    print(f"   Fake      : {df['label'].sum()}")
    print(f"   Authentic : {(df['label'] == 0).sum()}")
# Finalized data generation parameters
