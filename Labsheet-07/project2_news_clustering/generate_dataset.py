"""Generates a synthetic news-article dataset (5 topics, 300 articles) -> news_articles.csv"""
import random, pandas as pd
random.seed(7)
topics = {
 "sports": ("match team captain coach tournament league goal wicket stadium players season championship score victory defeat trophy final bowler striker innings medal referee fans".split(),
            ["The {a} won the {b} after a thrilling {c}.", "Fans filled the {a} to watch the {b} {c}.", "The {a} praised the {b} following the {c} of the season."]),
 "technology": ("software artificial intelligence startup smartphone cloud processor algorithm chip app developers cybersecurity robot network data launch update internet platform gadget semiconductor model".split(),
            ["A new {a} {b} was unveiled with improved {c}.", "Engineers released a {a} update focused on {b} and {c}.", "The company announced its {a} {b} for {c}."]),
 "politics": ("government minister parliament election opposition policy voters party senate bill campaign cabinet law reform prime president coalition debate manifesto vote".split(),
            ["The {a} introduced a {b} in {c}.", "Leaders of the {a} debated the {b} before the {c}.", "The {a} criticised the new {b} ahead of the {c}."]),
 "business": ("market stocks inflation investors company profit shares bank revenue economy trade merger earnings growth quarter dollar budget exports industry sales".split(),
            ["The {a} reported strong {b} this {c}.", "Analysts expect {a} {b} to affect the {c}.", "Rising {a} pushed {b} higher in the {c}."]),
 "health": ("hospital doctors patients vaccine disease treatment medicine virus clinical research symptoms surgery nutrition cancer therapy wellness diagnosis health study drug".split(),
            ["Doctors at the {a} began a {b} for {c}.", "A new {a} study links {b} with {c}.", "Researchers developed a {a} {b} to improve {c}."]),
}
generic = "said report according officials new year people week city country announced reportedly today yesterday sources".split()
rows = []
for t, (words, templates) in topics.items():
    for _ in range(60):
        sents = []
        for _ in range(random.randint(3, 5)):
            tpl = random.choice(templates)
            sents.append(tpl.format(a=random.choice(words), b=random.choice(words), c=random.choice(words)))
        other = random.choice([k for k in topics if k != t])
        noise = random.sample(topics[other][0], random.randint(0, 2)) + random.sample(generic, 3)
        sents.append("Officials " + " ".join(noise) + ".")
        rows.append({"text": " ".join(sents), "true_category": t})
df = pd.DataFrame(rows).sample(frac=1, random_state=5).reset_index(drop=True)
df.insert(0, "id", range(1, len(df) + 1))
df.to_csv("news_articles.csv", index=False); print(df.shape)
