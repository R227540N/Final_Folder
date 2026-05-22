from flask import Flask
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
import re

app = Flask(__name__)

def clean_text(text):
    """Removes HTML tags from summaries for cleaner display"""
    return re.sub(r'<[^>]+>', '', text)

@app.route('/')
def home():
    FEEDS = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss",
        "https://moxie.foxnews.com/google-publisher/world.xml"
    ]

    articles = []
    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:20]:  # 20 per feed keeps it fast for Vercel's limits
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": clean_text(entry.get("summary", "")[:150] + "..."),
                "source": feed.feed.get("title", "News Source")
            })

    if not articles:
        return "Error fetching news.", 500

    # 1. Prepare Data
    texts = [f"{article['title']} {article['summary']}" for article in articles]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)

    # 2. Cluster Articles
    num_clusters = min(8, len(articles) // 6) 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # 3. Extract Topic Names (Top 3 Keywords per cluster)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_topics = {}
    for i in range(num_clusters):
        top_terms = [terms[ind].capitalize() for ind in order_centroids[i, :3]]
        cluster_topics[i] = " • ".join(top_terms)

    # Group articles
    clustered_data = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clustered_data[label].append(articles[i])

    # 4. Generate the Modern UI
    current_time = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live News Topics</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b; --accent: #3b82f6; --border: #e2e8f0; }}
            * {{ box-sizing: border-box; }}
            body {{ font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 0; line-height: 1.5; }}
            header {{ background: var(--card); padding: 40px 20px; text-align: center; border-bottom: 1px solid var(--border); margin-bottom: 40px; }}
            h1 {{ margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.05em; }}
            .subtitle {{ color: var(--muted); margin-top: 10px; font-size: 1rem; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 0 20px 60px; display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 30px; }}
            .cluster-card {{ background: var(--card); border-radius: 16px; padding: 24px; border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s; }}
            .cluster-card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border-color: var(--accent); }}
            .topic-header {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); font-weight: 600; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #f1f5f9; }}
            .article {{ margin-bottom: 20px; }}
            .article:last-child {{ margin-bottom: 0; }}
            .article a {{ text-decoration: none; color: var(--text); font-weight: 600; font-size: 1.1rem; line-height: 1.4; display: block; margin-bottom: 8px; transition: color 0.2s; }}
            .article a:hover {{ color: var(--accent); }}
            .summary {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 10px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
            .meta {{ display: flex; align-items: center; gap: 10px; font-size: 0.75rem; color: var(--muted); font-weight: 600; }}
            .source-tag {{ background: #f1f5f9; padding: 4px 10px; border-radius: 20px; color: #475569; }}
        </style>
    </head>
    <body>
        <header>
            <h1>Global News Radar</h1>
            <div class="subtitle">Live AI-Clustered Topics • Updated {current_time}</div>
        </header>
        <div class="container">
    """

    for cluster_id, cluster_articles in clustered_data.items():
        if len(cluster_articles) < 2: 
            continue # Skip clusters that only have 1 lonely article
            
        topic_name = cluster_topics[cluster_id]
        
        html += f'<div class="cluster-card">\n'
        html += f'<div class="topic-header">{topic_name}</div>\n'
        
        for article in cluster_articles:
            html += f'''
            <div class="article">
                <a href="{article['link']}" target="_blank">{article['title']}</a>
                <div class="summary">{article['summary']}</div>
                <div class="meta">
                    <span class="source-tag">{article['source']}</span>
                </div>
            </div>
            '''
        html += '</div>\n'

    html += """
        </div>
    </body>
    </html>
    """
    
    return html

# Vercel needs this to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)