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
    add cumumlatSplit dates for smallive endorsement score, considering negative and positive prior endorsements.
    
Process Twitter Feed:

    Pre-requisite:
        1. sedft.cmd
            Gets up-to-date tick data and merges it with cached data.
            Remember to rename the new file and constant to help cache the new stuff.
        2. sefs.cmd
            Gets up-to-date equity fundamentals and puts in splits-fundamentals folder.
        3. Manually download Sharadar Actions
        4. Tip Ranks
            In project blogger_high_score:
                Step i. python -m bhs.services.tip_ranks_service.py
                Step ii. Run 1-clean-raw.ipynb

    1. Download Twitter Daily Data
    1.5 Smallified
    2. Fix Tweet Multi Key (fixed_drop)
    3. Flatten
    4. Add Id
    5. Deduped
    6. Assign Sentiment
    7. Learning Prep
    8. Great Reduction (gathering all)
    9. Twitter ML (gathering all)
    
    # TODO: 2020-12-25: chris.flesche:
        Phase I Completion
            Backup files to thumb drive. Done. 
            Change pipes to not use parent folder if parent folder is "stage".
            Validate predicted purchase results with historical data. 
            Test all columns for correlative power.
            Test results by removing ranking.
            Test risk of using smaller buy lot size.
                n Per day
                n Per week
            Incorporate recent up-to-date Tweet data.
            Re-run models
            If results acceptable start phase II
        
        Phase II
            Backup to thumb drive.
            Disconnect all projects from remote origin
            Sign up for realtime quotes
            Sign up for tip-ranks (?)
                Download tip-rank data daily.
            Create process to fetch all tweets since last market close.
            Create process to get up-to-date market data (merge recent days with cached data).
            Create process to download yesterday's Sharadar actions (?)
            Create process to download yesterday's fundy data (?)
            Create service to fetch realtime quotes within 15 (less?) minutes of close. Use realtime quotes to fudge close data.
            Train model with fudged data
            Feed recent data to model to get predictions
    
    
Process Price Target:

    1. Download EOD dailies in stock_predictor.
    2. Execute create_tickers_available_on_day
