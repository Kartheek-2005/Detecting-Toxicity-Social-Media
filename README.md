<h1>Detecting Toxicity in Social Media</h1>
<p>Our architecture builds upon the <a href="https://huggingface.co/unitary/unbiased-toxic-roberta">unitary/unbiased-toxic-roberta</a> model with the following modifications to enhance its ability to detect toxicity in social media texts:</p>

<h3>1. Toxicity-Attended Attention Block</h3>
<ul>
    <li><strong>Queries</strong>: A curated list of profane words (e.g., “idiot,” “fucking,” “stupid,” etc.).</li>
    <li><strong>Keys & Values</strong>: Derived from the embeddings of the input text.</li>
</ul>
<p>This additional attention block uses profane words as queries to focus on regions of the input text that are most relevant to determining toxicity. By selectively amplifying the context surrounding these words, the model can make more informed predictions.</p>

<h3>2. Contextual Analysis via Attention</h3>
<p>Attention mechanisms excel at identifying relevant parts of the input for a given task. By passing swear words as queries, the model ensures its focus remains on parts of the input that may signal offensive language, even when it is expressed subtly or ambiguously.</p>
