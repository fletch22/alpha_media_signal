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
    2. Pipe Smallified
    ~~3. Pipe Fix Tweet Multi Key (fixed_drop)~~ # No long er necessary
    4. Flatten
    5. Pipe Add Id
    6. Pipe Deduped
    7. Pipe Assign Sentiment (old 2-asign-sentiment) 
    8. Pipe Learning Prep (old 3-twitter_learning_prep)
    9. Pipe Great Reduction (gathering all)
    10. Twitter ML (gathering all)
    
    # TODO: 2020-12-25: chris.flesche:
        Phase I Completion
            Backup files to thumb drive. Done. 
            Change pipes to not use parent folder if parent folder is "stage". Will not do.
            Validate predicted purchase results with historical data. 
                Status: Will validate with WTD (WorldTradingData for live predictions, historical for historical predictions.
            Test all columns for correlative power.
                Status:
                    SMA + EOD_of_Purchase_Day seems to be strong
                    
            Test results by removing ranking.
                Status:
            Test risk of using smaller buy lot size.
                n Per day
                n Per week
                Status: Done. 8 purchases seems good.
            Incorporate recent up-to-date Tweet data.
                Status: Ongoing.
            If results acceptable start phase II
            Fix prediction - why lower? Bad scaling? Early scaling? cat_uniques off/bugged? Consider saving the scaler with the model.
        
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

## Monte Carlo MA Bottom Feeder:

### Exploration
	
	1. Learn Equity Fund Service indicators - func 'explain_fundy_fields'.
	2. Get top n of some indicator 'A' from Equity Fund Service - func 'get_top_by_attribute'
	3. Take list of stocks by year, submit to ma_bottom_feeder get_roi
	4. Based on roi, select list of stocks.

### Investment
	
	1. Choose recent stock list from above.
	2. Get stock pick recommenations from ma_bottom_feeder func 'get_recommendations'