This codebase contains 2 Ml projects - tweet sentiments and google news sentiment

# Install

    Run:
        conda install --user --file requirements.txt
    Run:
        pip install -U -r requirements.txt
    Run:
        conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

## Google News Sentiment

The news page follows this layout:

div:ZINbbc xpd O9g5cc uUPGi
	div:kCrYT
		a:href (source_href)
		h3
			div:BNeawe vvjwJb AP7Wnd (title)
		div:BNeawe vvjwJb AP7Wnd (source)
	div
	div:kCrYT
		div
			div:BNeawe s3v9rd AP7Wnd
				div:BNeawe s3v9rd AP7Wnd
					span:r0bn4c rQMQod
					span:r0bn4c rQMQod
					<text> (synopsis)
					

Phase: BloggerHighScore
    do searches with twitter handle. 
        results? have dollar tag?
    get top 25 bloggers on TipRank - get Twitter handles.
    include NYSE
    scrape info from TipRanks - find twitter handles
    split files into small chunks, encode tweets so can save as csv
    use batchy_bae
    use Kafka
    calculate average stock movement
        S&P 500
        Large, Medium and Small cap
    add cumumlative endorsement score, considering negative and positive prior endorsements.
