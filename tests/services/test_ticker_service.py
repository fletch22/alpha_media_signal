import time
from datetime import datetime
from typing import Tuple

import pandas as pd

from ams.DateRange import DateRange
from ams.services import ticker_service
from ams.services.EquityFields import EquityFields
from ams.utils import date_utils

# NOTE: 2020-10-02: chris.flesche: vol > 100K; price > 5
ticker_good = ['A', 'AA', 'AAGIY', 'AAI', 'AAL', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AATI', 'AAWW', 'AAXN', 'AB', 'ABAX', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCD', 'ABCO', 'ABCW', 'ABDC', 'ABG', 'ABII', 'ABM', 'ABMD', 'ABR', 'ABT', 'ABVT', 'ACA', 'ACAD', 'ACAM', 'ACAP', 'ACAS', 'ACAT', 'ACC', 'ACCD', 'ACCL', 'ACCO', 'ACCYY', 'ACEL', 'ACET', 'ACF', 'ACGL', 'ACHC', 'ACHN', 'ACI', 'ACIA', 'ACIW', 'ACL', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ACO', 'ACOM', 'ACRE', 'ACS', 'ACSF', 'ACTC', 'ACTL1', 'ACTV', 'ACV1', 'ADAM', 'ADAP', 'ADBE', 'ADC', 'ADCT', 'ADI', 'ADM', 'ADNT', 'ADP', 'ADPI', 'ADPT', 'ADS', 'ADSK', 'ADSW', 'ADT', 'ADT1', 'ADTN', 'ADVM', 'ADVS', 'ADY', 'ADYEY', 'AEA', 'AEC', 'AEE', 'AEGN', 'AEIS', 'AEL', 'AEM', 'AEO', 'AEP', 'AEPI', 'AER', 'AERI', 'AES', 'AET', 'AEYE', 'AEZ', 'AF-PC', 'AF', 'AFAM', 'AFCO', 'AFFX', 'AFG', 'AFIB', 'AFIN', 'AFL', 'AFOP', 'AFSI-PA', 'AFSI-PB', 'AFSI-PD', 'AFSI-PE', 'AFSI-PF', 'AFYA', 'AG', 'AGAM', 'AGCO', 'AGI', 'AGIO', 'AGLE', 'AGN-PA', 'AGN', 'AGN1', 'AGNC', 'AGO', 'AGP', 'AGR', 'AGU', 'AHAC', 'AHCO', 'AHH', 'AHL', 'AHPI', 'AIG', 'AIMC', 'AIMT', 'AIN', 'AINV', 'AIPC', 'AIR', 'AIRM', 'AIRV', 'AIT', 'AIV', 'AIZ', 'AJG', 'AJRD', 'AJX', 'AKAM', 'AKCA', 'AKR', 'AKRO', 'AKTS', 'AL', 'ALB', 'ALBO', 'ALC', 'ALC1', 'ALDN', 'ALDR', 'ALDW', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALGN', 'ALGT', 'ALJ', 'ALK', 'ALKS', 'ALL-PH', 'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLT', 'ALLY-PA', 'ALLY', 'ALNY', 'ALOG', 'ALOY', 'ALR', 'ALRM', 'ALSN', 'ALT', 'ALTE', 'ALTG', 'ALTR', 'ALTR1', 'ALV', 'ALVR', 'ALXN', 'ALY', 'AM', 'AM1', 'AM2', 'AMAC', 'AMAG', 'AMAP', 'AMAT', 'AMBA', 'AMBC', 'AMBI', 'AMBR', 'AMCC', 'AMCI', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMG', 'AMGN', 'AMH-PB', 'AMH-PC', 'AMH', 'AMID', 'AMK', 'AMKR', 'AMLN', 'AMMD', 'AMN', 'AMN1', 'AMP', 'AMPH', 'AMRC', 'AMRE', 'AMRI', 'AMRK', 'AMSC', 'AMSG', 'AMSWA', 'AMT-PB', 'AMT', 'AMTD', 'AMTG', 'AMWD', 'AMWL', 'AMX', 'AMZN', 'AN', 'ANAB', 'ANAC', 'ANCX', 'ANDE', 'ANDV', 'ANDX', 'ANET', 'ANF', 'ANGI', 'ANGO', 'ANN', 'ANNX', 'ANSS', 'ANSW', 'ANTM', 'AOBC', 'AOL', 'AON', 'AOS', 'AOUT', 'APA', 'APAC', 'APAGF', 'APAM', 'APC', 'APD', 'APD.W', 'APDN', 'APFH', 'APG', 'APH', 'API', 'APIC', 'APKT', 'APL', 'APLE', 'APLP', 'APLS', 'APLT', 'APO', 'APOG', 'APOL', 'APPF', 'APPN', 'APPS', 'APRN', 'APSG', 'APT', 'APTI', 'APTO', 'APTS', 'APTV', 'APU', 'APXT', 'APY', 'APY.W', 'AQ', 'AQ1', 'AQN', 'AQUA', 'ARA1', 'ARB', 'ARBA', 'ARCB', 'ARCC', 'ARCE', 'ARCH', 'ARCL', 'ARCT', 'ARCT1', 'ARD1', 'ARDX', 'ARE-PD', 'ARE-PE', 'ARE', 'ARES', 'ARG', 'ARGO', 'ARGX', 'ARI', 'ARIA', 'ARII', 'ARLO', 'ARMH', 'ARMK', 'ARMO', 'ARNA', 'ARNC-PB', 'ARNC', 'AROC', 'ARPI', 'ARQL', 'ARQT', 'ARR', 'ARRS', 'ARRY', 'ARTC', 'ARTG', 'ARUN', 'ARVN', 'ARW', 'ARWR', 'ARX', 'ASB', 'ASCA', 'ASEI', 'ASGN', 'ASH', 'ASH.W', 'ASIA', 'ASIX', 'ASMB', 'ASML', 'ASND', 'ASPL', 'ASPM', 'ASPN', 'ASPU', 'ASPX', 'ASTE', 'ASTX', 'ASUR', 'ASX.W', 'ATAC1', 'ATCO', 'ATEC', 'ATEN', 'ATGE', 'ATH-PC', 'ATH', 'ATHA', 'ATHL', 'ATHM', 'ATHN', 'ATHR', 'ATI', 'ATKR', 'ATLS1', 'ATLS2', 'ATMI', 'ATML', 'ATN', 'ATNM', 'ATNX', 'ATO', 'ATOM', 'ATR', 'ATRA', 'ATRC', 'ATRO', 'ATSG', 'ATU', 'ATUS', 'ATVI', 'ATW', 'AU', 'AUB', 'AUDC', 'AUPH', 'AUTH', 'AUVI', 'AUXL', 'AUY', 'AV', 'AVA', 'AVAV', 'AVB', 'AVD', 'AVDL', 'AVDR', 'AVEO', 'AVGO', 'AVHI', 'AVID', 'AVIV', 'AVLR', 'AVNR', 'AVNS', 'AVNT', 'AVOL', 'AVP', 'AVRO', 'AVT', 'AVTR', 'AVX', 'AVY', 'AVYA', 'AWAY', 'AWHHF', 'AWI', 'AWK', 'AWR', 'AX', 'AXDX', 'AXE', 'AXGN', 'AXL', 'AXLL', 'AXNX', 'AXP', 'AXS', 'AXSM', 'AXTA', 'AXTI', 'AXYS', 'AY', 'AYE', 'AYI', 'AYR', 'AYX', 'AZEK', 'AZN', 'AZO', 'AZPN', 'AZRE', 'AZUL', 'AZZ', 'B', 'BA', 'BABA', 'BABY', 'BAC-PM', 'BAC', 'BAC.WSA', 'BACHY', 'BAESY', 'BAGL', 'BAH', 'BAM', 'BANC', 'BAND', 'BAP', 'BASFY', 'BATRK', 'BATS', 'BAX', 'BAX.W', 'BAYRY', 'BBBB', 'BBBY', 'BBIO', 'BBL', 'BBT-PD', 'BBT-PE', 'BBT', 'BBY', 'BC', 'BCAT', 'BCC', 'BCE', 'BCEI', 'BCEL', 'BCLI', 'BCO', 'BCOR', 'BCOV', 'BCR', 'BCSF', 'BCSI', 'BDBD', 'BDC', 'BDK', 'BDN', 'BDTX', 'BDX', 'BDXA', 'BE', 'BEAM', 'BEAT', 'BEAV', 'BEC', 'BECN', 'BEE', 'BEEM', 'BEKE', 'BEL', 'BELM', 'BEN', 'BEP', 'BEPC', 'BERY', 'BETR', 'BEXP', 'BEZ', 'BF.B', 'BFAM', 'BFT', 'BFYT', 'BG', 'BGC', 'BGFV', 'BGG', 'BGNE', 'BGS', 'BHBK', 'BHC', 'BHE', 'BHF', 'BHGE', 'BHI', 'BHLB', 'BHP', 'BHS', 'BHVN', 'BID', 'BIDU', 'BIG', 'BIGC', 'BIIB', 'BILI', 'BILL', 'BIP-PA', 'BIP', 'BIPC', 'BIRT', 'BITA', 'BIVV', 'BJ', 'BJ1', 'BJGP', 'BJRI', 'BJS', 'BK', 'BKC1', 'BKE', 'BKH', 'BKI', 'BKI1', 'BKMU', 'BKNG', 'BKR', 'BKS', 'BKS.W', 'BKU', 'BKW', 'BKYF', 'BL', 'BLC', 'BLD', 'BLD.W', 'BLDP', 'BLDR', 'BLFS', 'BLI', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLMT', 'BLNK', 'BLOX', 'BLT', 'BLUD', 'BLUE', 'BLX', 'BMA', 'BMC', 'BMCH', 'BMI', 'BMO', 'BMR', 'BMRG', 'BMRN', 'BMS', 'BMTI', 'BMY', 'BNCL', 'BNCN', 'BNE', 'BNFT', 'BNI', 'BNK', 'BNL', 'BNNY', 'BNPQY', 'BNR', 'BNS', 'BNTX', 'BOBE', 'BOH', 'BOJA', 'BOKF', 'BONA', 'BOOT', 'BOWX', 'BOX', 'BOX1', 'BP', 'BPFH', 'BPL', 'BPMC', 'BPMP', 'BPOP', 'BPR', 'BPY', 'BPYU', 'BR', 'BRBR', 'BRC', 'BRCD', 'BRCM', 'BRDR', 'BRE', 'BREW', 'BRG', 'BRK.B', 'BRKL', 'BRKR', 'BRKS', 'BRMK', 'BRNC', 'BRO', 'BRP', 'BRPM', 'BRSS', 'BRX', 'BRY1', 'BSAC', 'BSIG', 'BSM', 'BSX', 'BSY', 'BT', 'BTAI', 'BTG', 'BTI', 'BTTGY', 'BUCY', 'BUD', 'BUFF', 'BURL', 'BUSE', 'BV', 'BV1', 'BVN', 'BWA', 'BWP', 'BWXT', 'BWY1', 'BX', 'BXG1', 'BXLT', 'BXMT', 'BXP', 'BXS', 'BYD', 'BYDDY', 'BYI', 'BYND', 'BZ', 'BZH', 'BZUN', 'C-PK', 'C-PN', 'C', 'CA', 'CAA', 'CAB', 'CACB', 'CACC', 'CACI', 'CACQ', 'CADE', 'CADX', 'CAE', 'CAFD', 'CAG', 'CAG.W', 'CAH', 'CAI', 'CAJ', 'CAKE', 'CAL', 'CAL1', 'CALD', 'CALL', 'CALM', 'CALP', 'CALX', 'CAM', 'CAMB', 'CAMP', 'CAPR', 'CAR', 'CARA', 'CARG', 'CARO', 'CARR', 'CARS', 'CARV', 'CASC1', 'CASH', 'CASY', 'CAT', 'CATB', 'CATM', 'CATO', 'CATT', 'CATY', 'CAVM', 'CB', 'CB1', 'CBAY', 'CBB', 'CBD', 'CBE', 'CBF', 'CBI', 'CBLK', 'CBM', 'CBNJ', 'CBOE', 'CBOU', 'CBPO', 'CBPX', 'CBRE', 'CBRL', 'CBS', 'CBSH', 'CBST', 'CBT', 'CBU', 'CBZ', 'CC', 'CCAC', 'CCC', 'CCC1', 'CCE1', 'CCEP', 'CCG', 'CCH', 'CCI-PA', 'CCI', 'CCIV', 'CCJ', 'CCK', 'CCL', 'CCMP', 'CCOI', 'CCP', 'CCP.W', 'CCRN', 'CCS', 'CCSC', 'CCT', 'CCU', 'CCX', 'CCXI', 'CCXX', 'CDAY', 'CDE', 'CDI', 'CDK', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDW', 'CDXS', 'CE', 'CEB', 'CEC', 'CECO', 'CEG', 'CELG', 'CELH', 'CELL', 'CENTA', 'CENX', 'CEO', 'CEPH', 'CEQP-P', 'CEQP', 'CERN', 'CERS', 'CF', 'CFFN', 'CFG', 'CFII', 'CFL', 'CFN', 'CFNL', 'CFR', 'CFRX', 'CFSG', 'CFX', 'CG', 'CGBD', 'CGC', 'CGEN', 'CGLD1', 'CGNX', 'CGRO', 'CGX', 'CHCT', 'CHD', 'CHDN', 'CHDX', 'CHEF', 'CHFC', 'CHFN', 'CHFN1', 'CHG', 'CHGG', 'CHH', 'CHIC1', 'CHKP', 'CHL', 'CHMI', 'CHMT', 'CHNG', 'CHPM', 'CHRD', 'CHRS', 'CHRS1', 'CHRW', 'CHSI', 'CHSP', 'CHT', 'CHTP', 'CHTR', 'CHU', 'CHUBA', 'CHUBK', 'CHUY', 'CHWY', 'CHX', 'CI', 'CIB', 'CIC', 'CICHY', 'CIEN', 'CIGI', 'CIM', 'CIMT', 'CINF', 'CIO', 'CISN', 'CIT', 'CITP', 'CIVI', 'CJ', 'CKEC', 'CKHUY', 'CKP', 'CKR', 'CKSW', 'CKXE', 'CL', 'CLB', 'CLBK', 'CLC', 'CLCD', 'CLDA', 'CLDR', 'CLDT', 'CLDX', 'CLF', 'CLFC', 'CLGX', 'CLH', 'CLI', 'CLLS', 'CLMS', 'CLNY1', 'CLP', 'CLR', 'CLS', 'CLSK', 'CLVS', 'CLX', 'CM', 'CMA', 'CMC', 'CMCSA', 'CMCSK', 'CMD', 'CME', 'CMG', 'CMGE', 'CMI', 'CML', 'CMLF', 'CMLP', 'CMLP1', 'CMO', 'CMP', 'CMPS', 'CMRE', 'CMS', 'CMTA', 'CMTL', 'CNA', 'CNC', 'CNCO', 'CNH', 'CNHI', 'CNI', 'CNK', 'CNL', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNQ', 'CNQR', 'CNR', 'CNSL', 'CNST', 'CNTY', 'CNU', 'CNVR', 'CNW', 'CNX', 'CNXM', 'COBZ', 'CODE', 'CODI', 'CODX', 'COF-PC', 'COF-PD', 'COF-PI', 'COF-PJ', 'COF-PK', 'COF-PP', 'COF', 'COG', 'COGT1', 'COHN', 'COHR', 'COHU', 'COL', 'COLB', 'COLD', 'COLE', 'COLL', 'COLM', 'COMM', 'CONE', 'CONN', 'COO', 'COOP', 'COP', 'COR', 'CORE', 'CORI', 'CORR', 'CORT', 'COST', 'COT', 'COTV', 'COUP', 'COV', 'COWN', 'CP', 'CPA', 'CPAA', 'CPB', 'CPD', 'CPGX', 'CPHD', 'CPHL', 'CPK', 'CPKI', 'CPL', 'CPLA', 'CPLG', 'CPN', 'CPNO', 'CPPL', 'CPRI', 'CPRT', 'CPS', 'CPT', 'CPTS', 'CPWM', 'CPWR', 'CPX', 'CPXX', 'CQB', 'CQH', 'CQP', 'CR', 'CRA', 'CRAY', 'CRBC', 'CRC', 'CRC.W', 'CRCM', 'CRDF', 'CRDN', 'CREE', 'CRH', 'CRHC', 'CRI', 'CRL', 'CRLBF', 'CRM', 'CRMD', 'CRN', 'CRNC', 'CRNCV', 'CRON', 'CROX', 'CRS', 'CRSA', 'CRSP', 'CRSR', 'CRTO', 'CRTX1', 'CRU', 'CRUS', 'CRV', 'CRWD', 'CRWN', 'CRY', 'CRZO', 'CS', 'CSBK', 'CSC', 'CSC.W', 'CSCD', 'CSCO', 'CSE', 'CSFL', 'CSGP', 'CSGS', 'CSH', 'CSII', 'CSIQ', 'CSL', 'CSOD', 'CSPR', 'CSR', 'CSRA', 'CST', 'CSTL', 'CSTM', 'CSX', 'CTAC', 'CTAS', 'CTB', 'CTCT', 'CTFO', 'CTL', 'CTLT', 'CTMX', 'CTRA', 'CTRE', 'CTRL', 'CTRN', 'CTRP', 'CTRX', 'CTS', 'CTSH', 'CTSO', 'CTT', 'CTV1', 'CTVA', 'CTWS', 'CTX1', 'CTXS', 'CUB', 'CUBE', 'CUBI', 'CUDA', 'CUE', 'CUK', 'CUNB', 'CURLF', 'CURO', 'CUTR', 'CUZ', 'CV', 'CVA', 'CVAC', 'CVBF', 'CVC', 'CVD', 'CVET', 'CVG', 'CVGI', 'CVGW', 'CVH', 'CVI', 'CVLT', 'CVM', 'CVNA', 'CVRR', 'CVS', 'CVT', 'CVTX', 'CVX', 'CW', 'CWEI', 'CWEN.A', 'CWEN', 'CWH', 'CWK', 'CWST', 'CWT', 'CXG', 'CXO', 'CXP', 'CXS', 'CXW', 'CYBE', 'CYBR', 'CYBS', 'CYBX', 'CYCL', 'CYCN', 'CYMI', 'CYN', 'CYNA', 'CYNI', 'CYNO', 'CYOU', 'CYPB', 'CYRX', 'CYS', 'CYT', 'CYTK', 'CZR', 'CZR2', 'CZZ', 'D', 'DAC', 'DADA', 'DAL', 'DAN', 'DANG', 'DANOY', 'DAO', 'DAR', 'DATA', 'DATE', 'DAVA', 'DB', 'DBD', 'DBI', 'DBTK', 'DBX', 'DCI', 'DCMYY', 'DCP', 'DCP1', 'DCPH', 'DCT', 'DCUC', 'DCUD', 'DD', 'DD1', 'DDC', 'DDIC', 'DDMX1', 'DDOG', 'DDRX', 'DDS', 'DDUP', 'DE', 'DEA', 'DECK', 'DEG', 'DEH', 'DEI', 'DEL', 'DELL', 'DELL.W', 'DELL1', 'DENN', 'DEO', 'DEP', 'DERM', 'DESP', 'DFG', 'DFIN', 'DFRG', 'DFS', 'DFT-PA', 'DFT', 'DG', 'DGI', 'DGII', 'DGIT', 'DGX', 'DHI', 'DHR', 'DHR.W', 'DHT', 'DIN', 'DIOD', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DK', 'DKNG', 'DKS', 'DLB', 'DLLR', 'DLM', 'DLPH', 'DLR-PG', 'DLR-PH-CL', 'DLR', 'DLTH', 'DLTR', 'DLX', 'DM', 'DMND', 'DMRC', 'DMYD', 'DMYT', 'DNA', 'DNB', 'DNBF', 'DNBK', 'DNKN', 'DNLI', 'DNNGY', 'DO', 'DOC', 'DOCU', 'DOLE1', 'DOMO', 'DOOR', 'DORM', 'DOTA', 'DOV', 'DOW', 'DOW.W', 'DOX', 'DOYU', 'DPHC', 'DPHCW', 'DPL', 'DPZ', 'DQ', 'DRC', 'DRD', 'DRE', 'DRH', 'DRI', 'DRII', 'DRIV1', 'DRNA', 'DRQ', 'DRTX', 'DRYS', 'DSCP', 'DSGX', 'DSKE', 'DSKY', 'DSPG', 'DSSI', 'DST', 'DT', 'DTE', 'DTIL', 'DTLK', 'DTPI', 'DTSI', 'DTV', 'DTV1', 'DUF', 'DUK', 'DUSA1', 'DVA', 'DVN', 'DWA', 'DWDP.W', 'DWRE', 'DX', 'DXC', 'DXCM', 'DY', 'DYAX', 'DYN', 'DYNC', 'E', 'EA', 'EAC1', 'EADSY', 'EAF', 'EAT', 'EB', 'EBAY', 'EBAYV', 'EBF', 'EBIX', 'EBON', 'EBR', 'EBS', 'EC', 'ECL', 'ECLP', 'ECOL', 'ECOM', 'ECPG', 'ECYT', 'ED', 'EDE', 'EDG', 'EDIT', 'EDR', 'EDU', 'EE', 'EEFT', 'EEP', 'EEQ', 'EF', 'EFC', 'EFII', 'EFX', 'EGAN', 'EGC', 'EGHT', 'EGL', 'EGN', 'EGO', 'EGOV', 'EGP', 'EGRX', 'EHC', 'EHIC', 'EHTH', 'EIGI', 'EIGR', 'EIX', 'EJ', 'EJPRY', 'EL', 'ELAN', 'ELAN.W', 'ELF', 'ELLI', 'ELMG', 'ELN', 'ELNK', 'ELOQ', 'ELP', 'ELRC', 'ELS', 'ELX', 'ELY', 'EM', 'EMC', 'EME', 'EMN', 'EMR', 'EMS', 'ENB', 'ENFC', 'ENH', 'ENIA', 'ENLK', 'ENOC', 'ENP', 'ENPH', 'ENR', 'ENR.W', 'ENS', 'ENSG', 'ENTA', 'ENTG', 'ENV', 'ENVA', 'ENVE', 'EOCA', 'EOG', 'EOPN', 'EP', 'EPAC', 'EPAM', 'EPAY', 'EPB', 'EPC', 'EPD', 'EPE1', 'EPHC', 'EPIQ', 'EPL', 'EPOC', 'EPR-PF', 'EPR', 'EPRT', 'EPZM', 'EQ', 'EQ1', 'EQC', 'EQGP', 'EQH', 'EQIX', 'EQM', 'EQNR', 'EQR', 'EQT', 'EQU', 'EQX', 'EQY', 'ERI', 'ERIC', 'ERII', 'ERT', 'ES', 'ESC', 'ESI', 'ESIC', 'ESIO', 'ESL', 'ESND', 'ESNT', 'ESPR', 'ESRT', 'ESRX', 'ESS', 'ESTC', 'ESV', 'ET', 'ET1', 'ETFC', 'ETH', 'ETM.W', 'ETN', 'ETNB', 'ETON', 'ETP-PE', 'ETP', 'ETP1', 'ETR', 'ETRN', 'ETRN.W', 'ETSY', 'EURN', 'EV', 'EVA', 'EVAC', 'EVBG', 'EVDY', 'EVER', 'EVER1', 'EVH', 'EVHC', 'EVOP', 'EVR', 'EVRG', 'EVRI', 'EVSI', 'EVTC', 'EVVV', 'EW', 'EWBC', 'EXAC', 'EXAM', 'EXAR', 'EXAS', 'EXC', 'EXEL', 'EXL', 'EXLS', 'EXP', 'EXPC', 'EXPD', 'EXPE', 'EXPI', 'EXPO', 'EXR', 'EYE', 'EYE2', 'EZCH', 'EZPW', 'F-PB', 'F', 'FACT', 'FAF', 'FANG', 'FANUY', 'FAST', 'FATE', 'FB', 'FBC', 'FBHS', 'FBK', 'FBM', 'FBNC', 'FBNK', 'FBP', 'FBR', 'FBRC', 'FC', 'FCAU', 'FCB', 'FCE.A', 'FCF', 'FCFP', 'FCFS', 'FCH-PA', 'FCH', 'FCL1', 'FCN', 'FCPT', 'FCS', 'FCX', 'FDC', 'FDML', 'FDO', 'FDP', 'FDS', 'FDUS', 'FDX', 'FE', 'FEAC', 'FEIC', 'FELE', 'FENC', 'FEYE', 'FF', 'FFBC', 'FFG', 'FFIC', 'FFIN', 'FFIV', 'FFWM', 'FG', 'FGEN', 'FGL', 'FGXI', 'FHB', 'FHI', 'FHN', 'FIATY', 'FIBK', 'FICO', 'FIF1', 'FIG', 'FII', 'FINL', 'FIO', 'FIRE', 'FIS', 'FISV', 'FIT', 'FITB', 'FIVE', 'FIVN', 'FIX', 'FIXX', 'FIZZ', 'FL', 'FLDM', 'FLEX', 'FLGT', 'FLIR', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT', 'FLTX', 'FLUX', 'FLWS', 'FLXN', 'FLXS', 'FLY', 'FMBI', 'FMC', 'FMCI', 'FMCIW', 'FMCN', 'FMER', 'FMR', 'FMS', 'FMSA', 'FMTX', 'FMX', 'FN', 'FNB', 'FND', 'FNDT', 'FNF', 'FNFG-PB', 'FNFG', 'FNKO', 'FNSR', 'FNV', 'FOCS', 'FOE', 'FOLD', 'FORM', 'FOSL', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FPO', 'FPU', 'FR', 'FRAC', 'FRC-PK', 'FRC', 'FREE', 'FRG', 'FRGI', 'FRHC', 'FRM', 'FRME', 'FRNK', 'FRO', 'FROG', 'FRP', 'FRPT', 'FRPT1', 'FRT', 'FRTA', 'FRX', 'FSB', 'FSCI', 'FSCT', 'FSII', 'FSIN', 'FSK', 'FSKR', 'FSL', 'FSLR', 'FSLY', 'FSM', 'FSR', 'FSS', 'FSYS', 'FTAC', 'FTAI', 'FTCH', 'FTDR', 'FTI', 'FTI1', 'FTLK', 'FTNT', 'FTO', 'FTRPR', 'FTS', 'FTSV', 'FTV', 'FTV.W', 'FUBC', 'FUL', 'FULT', 'FUN', 'FUR', 'FURX', 'FUTU', 'FUV', 'FVAC', 'FVRR', 'FWLT', 'FWONK', 'FWRD', 'G', 'GA', 'GAIN', 'GAME', 'GAN', 'GAS', 'GAS1', 'GATX', 'GBCI', 'GBDC', 'GBIO', 'GBNK', 'GBT', 'GBTC', 'GBX', 'GCAP', 'GCI.W', 'GCO', 'GCOM', 'GCP', 'GCP.W', 'GD', 'GDDY', 'GDEN', 'GDI', 'GDI1', 'GDOT', 'GDRX', 'GDS', 'GDYN', 'GE', 'GE.W', 'GEDU', 'GEF', 'GENZ', 'GEO', 'GEOI', 'GEOY', 'GES', 'GETI', 'GEVA', 'GFF', 'GFI', 'GFIG', 'GFL', 'GG', 'GGA', 'GGAL', 'GGG', 'GGP', 'GH', 'GHDX', 'GHIV', 'GIB', 'GIG', 'GIII', 'GIK', 'GIL', 'GILD', 'GIMO', 'GIS', 'GK', 'GKOS', 'GL', 'GLAC', 'GLAD', 'GLBC', 'GLBL', 'GLBL1', 'GLDC', 'GLDD', 'GLF', 'GLIBA', 'GLNG', 'GLOB', 'GLP', 'GLPG', 'GLPI', 'GLS', 'GLT', 'GLUU', 'GLW', 'GM', 'GM.WSA', 'GM.WSB', 'GMAB', 'GMCR', 'GME', 'GMED', 'GMHI', 'GMRE', 'GMS', 'GNBC', 'GNCMA', 'GNFT', 'GNL', 'GNMK', 'GNRC', 'GNRT', 'GNSS', 'GNTX', 'GO', 'GOCO', 'GOGO', 'GOL', 'GOLD', 'GOLF', 'GOOD', 'GOOG', 'GOOGL', 'GOOS', 'GOSS', 'GP', 'GPC', 'GPI', 'GPK', 'GPMT', 'GPN', 'GPRE', 'GPRO1', 'GPS', 'GPT', 'GPT2', 'GR', 'GRA', 'GRAF', 'GRAF.WS', 'GRAY', 'GRB', 'GRBK', 'GRFS', 'GRM', 'GRMN', 'GRPN', 'GRSV', 'GRT', 'GRUB', 'GRVY', 'GRWG', 'GS-PD', 'GS', 'GSBD', 'GSIC', 'GSK', 'GSL', 'GSM1', 'GSX', 'GT', 'GTBIF', 'GTES', 'GTHX', 'GTI', 'GTIV', 'GTLS', 'GTN', 'GTS', 'GTT', 'GTX.W', 'GTY', 'GUID', 'GVA', 'GWAY', 'GWB', 'GWPH', 'GWR', 'GWRE', 'GWW', 'GXGX', 'GXP-PB', 'GXP', 'GYMB', 'H', 'HA', 'HABT', 'HAE', 'HAFC', 'HAIN', 'HAL', 'HALO', 'HAR', 'HARP', 'HAS', 'HASI', 'HBAN', 'HBI', 'HBNC', 'HCA', 'HCAC', 'HCAT', 'HCBK', 'HCC', 'HCC1', 'HCCO', 'HCKT', 'HCM', 'HCP', 'HCP.W', 'HCSG', 'HCT', 'HD', 'HDB', 'HDNG', 'HDP', 'HDS', 'HE', 'HEAR', 'HEES', 'HEI.A', 'HEI', 'HELE', 'HEOP', 'HEP', 'HES-PA', 'HES', 'HESM', 'HEW', 'HF', 'HFC', 'HFFC', 'HFWA', 'HGEN', 'HGRD', 'HGSI', 'HGV', 'HGV.W', 'HHC', 'HHS', 'HI', 'HIBB', 'HIFR', 'HIG', 'HII', 'HIIQ', 'HILL', 'HITK', 'HITT', 'HIW', 'HK1', 'HL', 'HLF', 'HLI', 'HLIT', 'HLNE', 'HLSS', 'HLT', 'HLT.W', 'HLTH1', 'HMA', 'HMC', 'HME', 'HMI', 'HMIN', 'HMN', 'HMR', 'HMST', 'HMSY', 'HMY', 'HNBC', 'HNGR', 'HNH', 'HNHPF', 'HNR', 'HNT', 'HNZ', 'HOG', 'HOGS', 'HOLI', 'HOLX', 'HOMB', 'HOME', 'HON', 'HONE', 'HOPE', 'HOT', 'HOT.W', 'HOTT', 'HOV', 'HP', 'HPE', 'HPP', 'HPQ', 'HPQ.W', 'HPT', 'HPTX', 'HPY', 'HQCL', 'HQY', 'HR', 'HRB', 'HRBN', 'HRC', 'HRI', 'HRL', 'HRMY', 'HRTX', 'HRZN', 'HS', 'HSAC', 'HSBC-PA', 'HSBC', 'HSC', 'HSH', 'HSIC', 'HSII', 'HSNI', 'HSP', 'HST', 'HSTM', 'HSY', 'HT', 'HTA', 'HTBK', 'HTE', 'HTGC', 'HTH', 'HTHT', 'HTLD', 'HTM', 'HTS', 'HTSI', 'HTWR', 'HUBB', 'HUBG', 'HUBS', 'HUD', 'HUM', 'HUN', 'HURN', 'HUSI-PF', 'HUYA', 'HVB', 'HVT', 'HW', 'HWC', 'HWM', 'HXL', 'HYAC', 'HYAC1', 'HYC', 'HYMC', 'HYYDF', 'HZNP', 'HZO', 'I', 'IAA', 'IAA.W', 'IAC', 'IART', 'IBCA', 'IBEX', 'IBKC', 'IBKR', 'IBM', 'IBN', 'IBOC', 'IBP', 'IBTX', 'ICAD', 'ICE', 'ICHR', 'ICLK', 'ICLR', 'ICO', 'ICOC', 'ICPT', 'ICXT', 'ID', 'IDA', 'IDC', 'IDCBY', 'IDEV1', 'IDIX', 'IDTI', 'IDXX', 'IEA', 'IEX', 'IFF', 'IFJPY', 'IGMS', 'IGT', 'IGT1', 'IGTE', 'IHG', 'IHRT', 'IHS', 'IIIV', 'IIPR', 'IIVI', 'IL', 'ILG', 'ILMN', 'ILPT', 'IM', 'IMAX', 'IMDZ', 'IMMR', 'IMMU', 'IMO', 'IMPR', 'IMPV', 'IMS', 'IMTX', 'IMUX', 'IMVT', 'IMXI', 'IN', 'INCY', 'INDB', 'INET', 'INFA', 'INFN', 'INFO', 'INFU', 'INFY', 'ING', 'INGN', 'INGR', 'INHX', 'ININ', 'INMB', 'INMD', 'INN', 'INO', 'INOV', 'INSG', 'INSM', 'INSP', 'INST', 'INSU', 'INSW', 'INT', 'INTC', 'INTU', 'INVA', 'INVE', 'INVH', 'INVN', 'INXN', 'INZY', 'IOC', 'IONS', 'IOTS', 'IOVA', 'IP', 'IPAR', 'IPCC', 'IPCM', 'IPCR', 'IPCS', 'IPG', 'IPGP', 'IPHI', 'IPI', 'IPOA', 'IPOB', 'IPOB.WS', 'IPOC', 'IPSU', 'IPXL', 'IQ', 'IQNT', 'IQV', 'IR', 'IRBT', 'IRC', 'IRCP', 'IRDM', 'IRF', 'IRM', 'IRT', 'IRTC', 'IRWD', 'ISBC', 'ISCA', 'ISEE', 'ISIL', 'ISLE', 'ISNPY', 'ISRG', 'ISSI', 'ISYS', 'IT', 'ITC', 'ITCI', 'ITG', 'ITGR', 'ITLN', 'ITMN', 'ITOS', 'ITRI', 'ITT', 'ITW', 'ITWO', 'IUSA', 'IVC', 'IVTY', 'IVZ', 'IWA', 'IWOV', 'IXYS', 'J', 'JACK', 'JAH', 'JAMF', 'JAS', 'JASO', 'JAVA', 'JAZZ', 'JBGS', 'JBHT', 'JBL', 'JBLU', 'JBT', 'JCAP', 'JCG', 'JCI', 'JCI.W', 'JCOM', 'JD', 'JDAS', 'JE', 'JEC', 'JEF', 'JEF1', 'JELD', 'JHG', 'JIH', 'JIVE', 'JKHY', 'JKS', 'JLL', 'JMBA', 'JMG', 'JMI', 'JMIA', 'JNCE', 'JNJ', 'JNPR', 'JNS', 'JNY', 'JOBS', 'JOE', 'JOSB', 'JOY', 'JPM-PA', 'JPM-PC', 'JPM-PD', 'JPM-PE', 'JPM-PF', 'JPM-PH', 'JPM', 'JPM.WS', 'JRJC', 'JRN', 'JRVR', 'JST', 'JUNO', 'JW.A', 'JWN', 'JWS', 'K', 'KALA', 'KALU', 'KAMN', 'KANG', 'KAR', 'KAR.W', 'KATE', 'KBH', 'KBR', 'KBW', 'KC', 'KCAC', 'KCG', 'KCI', 'KCLI', 'KCP', 'KDDIY', 'KDN', 'KDP', 'KEI', 'KEM', 'KEN.W', 'KEX', 'KEY', 'KEYS', 'KEYW', 'KFN', 'KFX', 'KFY', 'KG', 'KGC', 'KH', 'KHC', 'KIDS', 'KIM-PH', 'KIM', 'KING', 'KIRK', 'KKD', 'KKR-PC', 'KKR', 'KL', 'KLAC', 'KLDO', 'KLIC', 'KLXI', 'KMB', 'KMB.W', 'KMDA', 'KMG', 'KMI-PA', 'KMI', 'KMP', 'KMPR', 'KMR', 'KMT', 'KMX', 'KN', 'KND', 'KNDI', 'KNDL', 'KNL', 'KNOL', 'KNSA', 'KNSL', 'KNSY', 'KNX', 'KNX1', 'KNXA', 'KO', 'KOD', 'KODK', 'KOF', 'KOG', 'KOP', 'KPTI', 'KR', 'KRA', 'KRC', 'KREF', 'KRFT', 'KRG', 'KRMD', 'KRNT', 'KRNY', 'KRO', 'KRP', 'KRTX', 'KRYS', 'KS', 'KSS', 'KSU', 'KT', 'KTB', 'KTB.W', 'KTOS', 'KTWO', 'KURA', 'KW', 'KWR', 'KYTH', 'KZ', 'L', 'LABL', 'LAC', 'LAD', 'LADR', 'LAKE', 'LAMR', 'LASR', 'LAUR', 'LAVA', 'LAYN', 'LAZ', 'LB', 'LB1', 'LBAI', 'LBRDK', 'LBRT', 'LBTYA', 'LBTYK', 'LCA', 'LCC', 'LCI', 'LCII', 'LCRD', 'LCRY', 'LDL', 'LDOS', 'LDR', 'LDRH', 'LDSH', 'LE', 'LEA', 'LECO', 'LEG', 'LEGN', 'LEN.B', 'LEN', 'LEVI', 'LEXEA', 'LFAC', 'LFC', 'LGF.A', 'LGF.B', 'LGIH', 'LGND', 'LH', 'LHCG', 'LHO', 'LHX', 'LI', 'LIA', 'LIFE2', 'LIHR', 'LII', 'LILA', 'LILAK', 'LIN', 'LINC', 'LIND', 'LION', 'LIOX', 'LITE', 'LIVK', 'LIVN', 'LK', 'LKQ', 'LL', 'LLL', 'LLNW', 'LLTC', 'LLY', 'LM', 'LMCK', 'LMND', 'LMNS', 'LMNX', 'LMPX', 'LMT', 'LNC', 'LNCE', 'LNCR', 'LNG', 'LNKD', 'LNT', 'LNTH', 'LNY', 'LO', 'LOB', 'LOCK', 'LOCO', 'LOGC', 'LOGI', 'LOGM', 'LOJN', 'LOOP1', 'LOPE', 'LOVE', 'LOW', 'LPDX', 'LPG', 'LPI', 'LPL', 'LPLA', 'LPNT', 'LPRO', 'LPS', 'LPSN', 'LPT', 'LPX', 'LQ', 'LQDT', 'LRCX', 'LRN', 'LSAC', 'LSCC', 'LSE', 'LSF', 'LSI', 'LSI1', 'LSPD', 'LSTR', 'LSXMA', 'LSXMK', 'LSXMR', 'LTC', 'LTHM', 'LTM', 'LTM1', 'LTRX', 'LTXB', 'LUFK', 'LULU', 'LUMN', 'LUNMF', 'LUV', 'LUX', 'LVB', 'LVGO', 'LVLT', 'LVMUY', 'LVS', 'LW', 'LWSN', 'LX', 'LXFR', 'LXFT', 'LXK', 'LXP', 'LYB', 'LYFT', 'LYTS', 'LYV', 'LZ', 'LZB', 'M', 'MA', 'MAA', 'MAC', 'MAG', 'MAIN', 'MAKO', 'MAN', 'MANH', 'MANT', 'MAPP', 'MAR', 'MAS', 'MAS.W', 'MASI', 'MAT', 'MATK', 'MATX', 'MAXN', 'MAXR', 'MB', 'MBFI', 'MBI', 'MBLY', 'MBOT', 'MBRG', 'MBT', 'MBUU', 'MC', 'MCCC', 'MCD', 'MCFT', 'MCHP', 'MCK', 'MCO', 'MCRB', 'MCRL', 'MCRN', 'MCRS', 'MCS', 'MCY', 'MD', 'MDAS', 'MDB', 'MDC', 'MDCO', 'MDF', 'MDGL', 'MDLA', 'MDLZ', 'MDMD', 'MDP', 'MDRX', 'MDS', 'MDSO', 'MDT', 'MDTH', 'MDU', 'ME', 'MED', 'MEDP', 'MEDW', 'MEDX', 'MEE', 'MEET', 'MEG', 'MEI', 'MELI', 'MEND', 'MENT', 'MEOH', 'MEP', 'MERC', 'MESG', 'MESO', 'MET-PB', 'MET-PF', 'MET', 'METR', 'MFB', 'MFC', 'MFE', 'MFN', 'MFRM', 'MFSF', 'MFW', 'MGA', 'MGAM', 'MGG', 'MGLN', 'MGM', 'MGNI', 'MGNX', 'MGP', 'MGPI', 'MGY', 'MHG', 'MHK', 'MHO', 'MHS', 'MI', 'MIC', 'MIDD', 'MIG', 'MIK', 'MIL1', 'MIME', 'MINI', 'MIPS', 'MIR', 'MIST', 'MITI', 'MITK', 'MITL', 'MJCO', 'MJN', 'MKC', 'MKSI', 'MKTG', 'MKTO', 'MKTX', 'MLAC', 'MLCO', 'MLHR', 'MLI', 'MLM', 'MLNX', 'MMC', 'MMDM', 'MMI', 'MMI1', 'MMM', 'MMP', 'MMR', 'MMS', 'MMSI', 'MMX', 'MMYT', 'MNOV', 'MNR', 'MNRK', 'MNRL', 'MNRO', 'MNST', 'MNT', 'MNTA', 'MO', 'MOBL', 'MOD', 'MODL', 'MODN', 'MOG.A', 'MOH', 'MOLX', 'MOMO', 'MON', 'MONT', 'MORE', 'MOS', 'MOV', 'MOVE', 'MPC', 'MPG', 'MPLX', 'MPO', 'MPR', 'MPS', 'MPW', 'MPWR', 'MR1', 'MRCY', 'MRD', 'MRGE1', 'MRH', 'MRK', 'MRNA', 'MRNS', 'MRSN', 'MRT', 'MRTN', 'MRTX', 'MRVL', 'MRX', 'MS-PK', 'MS', 'MSA', 'MSBI', 'MSCC', 'MSCI', 'MSFG', 'MSFT', 'MSG', 'MSGE', 'MSGN', 'MSGS', 'MSI', 'MSL', 'MSM', 'MSO', 'MSPD', 'MSWP', 'MT', 'MTA', 'MTB', 'MTCH', 'MTCH2', 'MTCR', 'MTDR', 'MTEC', 'MTEM', 'MTG', 'MTGE', 'MTH', 'MTN', 'MTOR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTW', 'MTX', 'MTZ', 'MU', 'MULE', 'MUR', 'MUSA', 'MUSA1', 'MV', 'MVL', 'MVNR', 'MWA', 'MWE', 'MWIV', 'MWK', 'MWV', 'MX', 'MXC', 'MXIM', 'MXL', 'MYCC', 'MYGN', 'MYL', 'MYOK', 'MYOV', 'MYRG', 'N', 'NABZY', 'NAL', 'NANO', 'NARI', 'NATI', 'NATL', 'NAV', 'NAVG', 'NAVI', 'NBBC', 'NBHC', 'NBIX', 'NBL', 'NBLX', 'NBR-PA', 'NBR', 'NBSE', 'NCFT', 'NCI', 'NCLH', 'NCNA', 'NCNO', 'NCOM', 'NCR', 'NCX', 'NDAQ', 'NDLS', 'NDN', 'NDRM', 'NDSN', 'NDZ', 'NEE-PI', 'NEE-PN', 'NEE-PQ', 'NEE-PR', 'NEE', 'NEFF', 'NEM', 'NEO', 'NEP', 'NET', 'NETE', 'NETL1', 'NEW', 'NEWM', 'NEWP', 'NEWR', 'NEWS', 'NFBK', 'NFC', 'NFE', 'NFG', 'NFIN', 'NFLX', 'NFP', 'NFX', 'NG', 'NGG', 'NGHC', 'NGLS', 'NGVC', 'NGVT', 'NHI', 'NHP', 'NHWK', 'NI', 'NICE', 'NILSY', 'NIO', 'NIU', 'NJR', 'NK', 'NKE', 'NKLA', 'NKLAW', 'NKTR', 'NKTX', 'NLC', 'NLOK', 'NLS', 'NLSN', 'NLTX', 'NLY-PA', 'NLY-PE', 'NLY-PF', 'NLY', 'NMBL', 'NMFC', 'NMIH', 'NMMC', 'NNN', 'NNOX', 'NOAH', 'NOC', 'NOG', 'NOMD', 'NORD', 'NOV', 'NOVA', 'NOVL', 'NOVN1', 'NOVS', 'NOW', 'NPA', 'NPBC', 'NPO', 'NPSP', 'NPTN', 'NRCG', 'NRDBY', 'NRE', 'NRE.W', 'NRF-PD', 'NRF', 'NRG', 'NRZ', 'NS', 'NSA', 'NSAM', 'NSANY', 'NSC', 'NSHA', 'NSIT', 'NSM', 'NSM1', 'NSP', 'NSR', 'NSRGY', 'NST', 'NSTC', 'NSTG', 'NTAP', 'NTB', 'NTCO', 'NTCT', 'NTDOY', 'NTES', 'NTG1', 'NTGR', 'NTI', 'NTK', 'NTLA', 'NTLS', 'NTNX', 'NTQ', 'NTR', 'NTRA', 'NTRI', 'NTRS', 'NTSP', 'NTST', 'NTTYY', 'NTUS', 'NTY', 'NUAN', 'NUE', 'NUHC', 'NUS', 'NUTR', 'NUVA', 'NVAX', 'NVCR', 'NVDA', 'NVDQ', 'NVE', 'NVLS1', 'NVMI', 'NVO', 'NVRO', 'NVS', 'NVST', 'NVST.W', 'NVT', 'NVT.W', 'NVTA', 'NWBI', 'NWE', 'NWHM', 'NWL', 'NWN', 'NWS', 'NWSA', 'NX', 'NXEO', 'NXGN', 'NXPI', 'NXRT', 'NXST', 'NXTC', 'NXTM', 'NXY', 'NYCB', 'NYRT', 'NYT', 'NYX', 'NZ', 'O', 'OA', 'OAC', 'OACB', 'OAK', 'OB', 'OC', 'OCAT', 'OCFC', 'OCFT', 'OCLR', 'OCR', 'OCUL', 'ODFL', 'ODP', 'ODSY', 'ODT', 'OEC', 'OESX', 'OFC', 'OFG', 'OFIX', 'OGE', 'OGS', 'OHI', 'OI', 'OILT', 'OKE', 'OKS', 'OKSB', 'OKTA', 'OLBK', 'OLED', 'OLLI', 'OLN', 'OM', 'OMC', 'OMCL', 'OME', 'OMER', 'OMF', 'OMG', 'OMI', 'OMN', 'OMP', 'OMPI', 'OMTR', 'OMX', 'ON', 'ONB', 'ONE2', 'ONEM', 'ONTO', 'ONVO', 'ONXX', 'OPCH', 'OPEN', 'OPES', 'OPI', 'OPLK', 'OPNT1', 'OPRA', 'OPRX', 'OPTR', 'OPWR', 'OPY', 'OR', 'ORA', 'ORAN', 'ORB', 'ORC', 'ORCC', 'ORCL', 'ORH', 'ORI', 'ORIC', 'ORIG', 'ORLY', 'OSB', 'OSGIQ', 'OSH', 'OSIP', 'OSIR', 'OSK', 'OSMT', 'OSPN', 'OSTK', 'OSUR', 'OSW', 'OTEX', 'OTIS', 'OTRK', 'OUT', 'OUTR', 'OVEN', 'OVID', 'OVTI', 'OVV', 'OWW', 'OXFD', 'OXM', 'OXY', 'OXY.W', 'OZK', 'OZM', 'P', 'PAA', 'PAAS', 'PAC', 'PACB', 'PACR', 'PACT', 'PACW', 'PAE', 'PAET', 'PAG', 'PAGP', 'PAGS', 'PAM', 'PAND', 'PANW', 'PAR', 'PAR1', 'PARR', 'PAS', 'PASG', 'PATK', 'PAY', 'PAYC', 'PAYS', 'PAYX', 'PB', 'PBA', 'PBCT', 'PBF', 'PBG', 'PBH', 'PBI', 'PBKS', 'PBR.A', 'PBR', 'PBTH', 'PBY1', 'PBYI', 'PCAR', 'PCBK', 'PCG', 'PCH', 'PCL', 'PCMI', 'PCP', 'PCRFY', 'PCRX', 'PCTY', 'PCVX', 'PCYC', 'PCZ', 'PD', 'PDAC', 'PDCE', 'PDCO', 'PDD', 'PDE', 'PDFS', 'PDGI', 'PDH', 'PDM', 'PE', 'PEAK', 'PEB', 'PECK', 'PEET', 'PEG', 'PEGA', 'PEGI', 'PEIX', 'PEN', 'PENN', 'PEP', 'PER1', 'PERI', 'PERY', 'PETM', 'PETQ', 'PETS', 'PF', 'PFCB', 'PFE', 'PFG', 'PFGC', 'PFLT', 'PFNX', 'PFPT', 'PFS', 'PFSI', 'PFSW', 'PG', 'PGEM', 'PGI', 'PGN1', 'PGND', 'PGNY', 'PGR', 'PGRE', 'PGTI', 'PH', 'PHG', 'PHH', 'PHI', 'PHM', 'PHR', 'PI', 'PIC', 'PII', 'PIKE', 'PINC', 'PING', 'PINS', 'PIPR', 'PIR', 'PJC', 'PK', 'PKE', 'PKG', 'PKI', 'PKT', 'PKX', 'PKY', 'PL', 'PLA', 'PLAB', 'PLAN', 'PLAY', 'PLCE', 'PLCM', 'PLD', 'PLD1', 'PLFE', 'PLKI', 'PLL', 'PLL1', 'PLMR', 'PLNR', 'PLNT', 'PLOW', 'PLT', 'PLUG', 'PLXS', 'PLXT', 'PLYM', 'PM', 'PMACA', 'PMC', 'PMCS', 'PMT', 'PMTI', 'PMVP', 'PNC-PP', 'PNC-PQ', 'PNC', 'PNFP', 'PNG', 'PNGAY', 'PNK', 'PNM', 'PNR', 'PNR.W', 'PNRA', 'PNW', 'PNY', 'PODD', 'POL', 'POM', 'POOL', 'POR', 'POST', 'POT', 'POWI', 'POWR', 'POZN', 'PPBI', 'PPC', 'PPD', 'PPDI', 'PPG', 'PPL', 'PPL.W', 'PPO', 'PPS', 'PQG', 'PRA', 'PRAA', 'PRAH', 'PRCP', 'PRDO', 'PRE', 'PRFT', 'PRGO', 'PRGS', 'PRI', 'PRIM', 'PRLB', 'PRLD', 'PRMW', 'PRMW1', 'PRNB', 'PRO', 'PROS', 'PROSY', 'PRPL', 'PRSC', 'PRSP', 'PRTA', 'PRTK', 'PRTS', 'PRU', 'PRVB', 'PRVL', 'PRX', 'PRXL', 'PS', 'PSA', 'PSAC', 'PSD', 'PSDO', 'PSE', 'PSEC', 'PSEM', 'PSMI', 'PSN', 'PSNL', 'PSO', 'PSS', 'PSSI', 'PSTB', 'PSTG', 'PSTH', 'PSTH.WS', 'PSTI', 'PSTX', 'PSX', 'PSXP', 'PSYS1', 'PTC', 'PTCT', 'PTGX', 'PTHN', 'PTLA', 'PTON', 'PTP', 'PTR', 'PTRY', 'PTV', 'PTVE', 'PUK', 'PVAC', 'PVG', 'PVG1', 'PVH', 'PVR', 'PVSW', 'PVT', 'PVTB', 'PVTL', 'PVX', 'PWER', 'PWR', 'PWRD', 'PX', 'PXD', 'PXP', 'PYPL', 'PYX', 'PZZA', 'Q1', 'QCOM', 'QCOR', 'QCP', 'QCP.W', 'QDEL', 'QEPM', 'QFIN', 'QGEN', 'QIHU', 'QIWI', 'QLGC', 'QLIK', 'QLTY', 'QLYS', 'QNST', 'QRE', 'QRTEA', 'QRTEP', 'QRVO', 'QSFT', 'QSR', 'QSR.W', 'QTNA', 'QTNT', 'QTRX', 'QTS', 'QTWO', 'QUNR', 'QUOT', 'QURE', 'R', 'RA2', 'RACE', 'RAD', 'RADA', 'RADS', 'RAH', 'RAI', 'RALY', 'RAMP', 'RAPT', 'RARE', 'RATE', 'RATE1', 'RAVN', 'RAX', 'RBA', 'RBAC', 'RBC', 'RBN', 'RBS-PE', 'RBS-PF', 'RBS-PG', 'RBS-PL', 'RBS', 'RC', 'RCI', 'RCII', 'RCKT', 'RCL', 'RCM', 'RCNI', 'RCPT', 'RCRC', 'RCUS', 'RDA', 'RDC', 'RDEA', 'RDEN', 'RDFN', 'RDHL', 'RDN', 'RDNT', 'RDS.A', 'RDS.B', 'RDUS', 'RDWR', 'RDY', 'RE', 'REAL', 'REG', 'REGI', 'REGN', 'REKR', 'RELX', 'REMY', 'REN', 'RENT', 'RENX', 'REPL', 'REPYY', 'RESI', 'RETA', 'REV', 'REVG', 'REXR', 'REYN', 'REZI', 'REZI.W', 'RF', 'RFMD', 'RGA', 'RGC', 'RGEN', 'RGLD', 'RGNX', 'RGP', 'RGR', 'RGS', 'RH', 'RHB', 'RHHBY', 'RHI', 'RHP', 'RHT', 'RIC', 'RICE', 'RIGP', 'RIO', 'RISK', 'RJF', 'RKT', 'RKUS', 'RL', 'RLAY', 'RLD', 'RLGT', 'RLGY', 'RLJ', 'RLMD', 'RLRN', 'RLYP', 'RMBL', 'RMBS', 'RMD', 'RMP', 'RNA', 'RNF', 'RNG', 'RNR', 'RNST', 'ROAD', 'ROC', 'ROCK', 'ROH', 'ROIC', 'ROK', 'ROKU', 'ROL', 'ROP', 'ROSE1', 'ROST', 'ROVI', 'RP', 'RPAI', 'RPAY', 'RPD', 'RPM', 'RPRX', 'RPT', 'RPTP', 'RRC', 'RRGB', 'RRMS', 'RRR', 'RRR1', 'RS', 'RSE', 'RSG', 'RSPP', 'RST', 'RSTI', 'RTEC', 'RTI', 'RTLR', 'RTN', 'RTP', 'RTRX', 'RTX', 'RUBI', 'RUBY', 'RUE', 'RUN', 'RURL', 'RUTH', 'RVBD', 'RVI1', 'RVLV', 'RVMD', 'RVNC', 'RVP', 'RWT', 'RX', 'RXDX', 'RXN-PA', 'RXN', 'RXT', 'RY', 'RYAAY', 'RYI', 'RYL', 'RYN', 'RYTM', 'S', 'SA', 'SAAS', 'SABR', 'SAFE', 'SAFM', 'SAGE', 'SAH', 'SAIA', 'SAIC', 'SAIL', 'SALE', 'SALT', 'SAMA', 'SAND', 'SANM', 'SAP', 'SAPE', 'SASR', 'SATS', 'SAVA', 'SAVE', 'SBAC', 'SBBX', 'SBCF', 'SBCP', 'SBE', 'SBGI', 'SBGL', 'SBH', 'SBIB', 'SBLK', 'SBNY', 'SBRA', 'SBRCY', 'SBS', 'SBSW', 'SBUX', 'SBX', 'SBY', 'SC', 'SCAI', 'SCCO', 'SCG', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SCLN', 'SCM', 'SCMP', 'SCNB', 'SCPL', 'SCS', 'SCU', 'SCVL', 'SDC', 'SDGR', 'SE', 'SE1', 'SEAS', 'SEDG', 'SEE', 'SEIC', 'SEM', 'SEMG', 'SEMI', 'SEND', 'SEP', 'SEPR', 'SERV', 'SES', 'SF', 'SFD', 'SFG', 'SFIX', 'SFL', 'SFLY', 'SFM', 'SFN1', 'SFNC', 'SFR', 'SFS', 'SFSF', 'SGBK', 'SGEN', 'SGH', 'SGI', 'SGMO', 'SGMS', 'SGNT', 'SGP', 'SGRY', 'SGY', 'SHAK', 'SHAW', 'SHEN', 'SHF', 'SHFL', 'SHG', 'SHLL', 'SHLL.WS', 'SHLM', 'SHLX', 'SHO', 'SHOO', 'SHOP', 'SHOR', 'SHP', 'SHS', 'SHW', 'SHYF', 'SIAL', 'SIEGY', 'SIFI', 'SIG', 'SIGA', 'SIGI', 'SILK', 'SILV', 'SIMG', 'SIMO', 'SINA', 'SIR', 'SIRI', 'SIRO', 'SITC', 'SITE', 'SITM', 'SIVB', 'SIX', 'SJI', 'SJM', 'SJR', 'SKIL', 'SKM', 'SKS', 'SKT', 'SKUL', 'SKX', 'SKY', 'SKYS', 'SKYW', 'SLAB', 'SLB', 'SLF', 'SLG', 'SLGN', 'SLH', 'SLM', 'SLP', 'SLQT', 'SLRC', 'SLXP', 'SMA', 'SMAR', 'SMCI', 'SMED', 'SMFG', 'SMG', 'SMOD', 'SMPL', 'SMSC', 'SMTC', 'SMTL', 'SMTS1', 'SNA', 'SNAP', 'SNBR', 'SNC', 'SNDA', 'SNDK', 'SNDR', 'SNDX', 'SNE', 'SNH', 'SNI', 'SNIC', 'SNN', 'SNOW', 'SNP', 'SNPR', 'SNPS', 'SNR.W', 'SNTS', 'SNV', 'SNWL', 'SNX', 'SNY', 'SO', 'SOA', 'SOAC', 'SOBKY', 'SODA', 'SOGO', 'SOHU', 'SOI', 'SON', 'SONA', 'SONE', 'SONO', 'SP', 'SPAQ', 'SPAR', 'SPB', 'SPB1', 'SPCE', 'SPEC', 'SPG', 'SPGI', 'SPH', 'SPI', 'SPIL', 'SPLK', 'SPLS', 'SPMD1', 'SPNC', 'SPNS', 'SPOK', 'SPOT', 'SPR', 'SPRD', 'SPRO', 'SPSC', 'SPT', 'SPTN', 'SPWH', 'SPWR', 'SPXC', 'SQ', 'SQBK', 'SQI', 'SQM', 'SQNS', 'SR', 'SRAC', 'SRC', 'SRC.W', 'SRCL', 'SRCLP', 'SRE', 'SRG', 'SRNE', 'SRPT', 'SRRK', 'SRX', 'SRZ', 'SSB', 'SSCC', 'SSD', 'SSL', 'SSNC', 'SSP', 'SSP.W', 'SSPK', 'SSRM', 'SSRX', 'SSSS', 'SSTK', 'SSW', 'SSYS', 'ST', 'STAA', 'STAG', 'STAR', 'STAY', 'STB', 'STBA', 'STBZ', 'STC', 'STE', 'STE1', 'STEC1', 'STEI', 'STEP', 'STI-PE', 'STI', 'STIM', 'STJ', 'STKL', 'STL', 'STL1', 'STLD', 'STM', 'STMP', 'STNE', 'STNG', 'STNR', 'STOR', 'STR', 'STRA', 'STRL', 'STRO', 'STRP', 'STRZA', 'STT', 'STWD', 'STX', 'STZ', 'SU', 'SUG', 'SUI', 'SUM', 'SUMO', 'SUN', 'SUN1', 'SUNH', 'SUPN', 'SUPX', 'SUR', 'SURF', 'SURG', 'SUSQ', 'SUSS', 'SUZ', 'SVAC', 'SVC', 'SVM', 'SVMK', 'SVR', 'SVU', 'SWAV', 'SWBI', 'SWC', 'SWCH', 'SWI', 'SWI1', 'SWIM', 'SWIR', 'SWK', 'SWKS', 'SWM', 'SWNC', 'SWP', 'SWS', 'SWSI', 'SWTX', 'SWWC', 'SWX', 'SWY', 'SXCP', 'SXE1', 'SXT', 'SY', 'SY2', 'SYA', 'SYF-PA', 'SYF', 'SYF.W', 'SYK', 'SYKE', 'SYMC', 'SYMM', 'SYNA', 'SYNH', 'SYNO', 'SYNT', 'SYRS', 'SYUT', 'SYY', 'T-PC', 'T', 'TA', 'TAC', 'TACO', 'TAK', 'TAL', 'TAL1', 'TALO', 'TAM', 'TAP', 'TAST', 'TATT', 'TAYC', 'TBBK', 'TBI', 'TBIO', 'TBL', 'TBPH', 'TBRA', 'TCBI', 'TCDA', 'TCEHY', 'TCF', 'TCF1', 'TCMD', 'TCNNF', 'TCO', 'TCOM', 'TCON', 'TCP', 'TCPC', 'TCRR', 'TCS', 'TD', 'TDC', 'TDG', 'TDOC', 'TDS', 'TDW', 'TDY', 'TE', 'TEA', 'TEAM', 'TEAM1', 'TECD', 'TECH', 'TECK', 'TEG', 'TEL', 'TEN', 'TENB', 'TEO', 'TEP', 'TER', 'TERP', 'TEVA', 'TEX', 'TFC-PR', 'TFC', 'TFFP', 'TFII', 'TFM', 'TFSL', 'TFX', 'TG', 'TGE', 'TGH', 'TGI', 'TGNA', 'TGP', 'TGT', 'TGTX', 'THBR', 'THC', 'THCB', 'THG', 'THI', 'THO', 'THOR1', 'THR', 'THRM', 'THS', 'TIBX', 'TIE', 'TIER', 'TIF', 'TIG', 'TIGO', 'TILE', 'TIN', 'TISI', 'TIVO1', 'TJX', 'TKLC', 'TKR', 'TKTM', 'TLCR', 'TLEO', 'TLK', 'TLMR', 'TLN', 'TLND', 'TLP', 'TLRA', 'TLRD', 'TLSMF', 'TM', 'TMCX', 'TMDX', 'TME', 'TMH', 'TMHC', 'TMK', 'TMO', 'TMS', 'TMUS', 'TMUSP', 'TNB', 'TNDM', 'TNET', 'TNGO', 'TNK', 'TNP', 'TNS', 'TOL', 'TOT', 'TOWN', 'TOWR', 'TPB', 'TPC', 'TPCG', 'TPGH', 'TPGI', 'TPH', 'TPIC', 'TPP', 'TPR', 'TPRE', 'TPTX', 'TPVG', 'TPX', 'TQNT', 'TR', 'TRA', 'TRAD', 'TRAK', 'TRCO', 'TRCR', 'TREX', 'TRGP', 'TRH', 'TRHC', 'TRI', 'TRIL', 'TRIP', 'TRK', 'TRLA1', 'TRLG', 'TRMB', 'TRMK', 'TRN', 'TRNE', 'TRNO', 'TRNX1', 'TROW', 'TROX', 'TRP', 'TRS', 'TRST', 'TRTL', 'TRTN', 'TRTX', 'TRU', 'TRUP', 'TRV', 'TRW', 'TRWH', 'TS', 'TSC', 'TSCDY', 'TSCO', 'TSE', 'TSEM', 'TSG', 'TSHA', 'TSL', 'TSLA', 'TSLX', 'TSM', 'TSN', 'TSRE', 'TSRO', 'TSRX', 'TSS', 'TST', 'TSU', 'TT', 'TTC', 'TTCMY', 'TTD', 'TTEK', 'TTES', 'TTGT', 'TTM', 'TTM.R', 'TTMI', 'TTWO', 'TU', 'TUBE', 'TUFN', 'TUMI', 'TUP', 'TUTR', 'TV', 'TVL', 'TVPT', 'TVTY', 'TW', 'TW2', 'TWB', 'TWC', 'TWCT', 'TWLL', 'TWLO', 'TWNK', 'TWO', 'TWOU', 'TWST', 'TWTC', 'TWTR', 'TX', 'TXG', 'TXI', 'TXN', 'TXRH', 'TXT', 'TXTR', 'TYC', 'TYL', 'TYPE', 'U', 'UA', 'UAA', 'UAL', 'UAM', 'UBA', 'UBER', 'UBNK', 'UBNK1', 'UBNT', 'UBS', 'UBSI', 'UCBI', 'UCTT', 'UDF', 'UDR', 'UDRL', 'UE', 'UFPI', 'UFS', 'UGI', 'UHS', 'UI', 'UIL', 'UIS', 'UL', 'ULTA', 'ULTI', 'UMBF', 'UMH', 'UMPQ', 'UN', 'UNFI', 'UNH', 'UNIT', 'UNM', 'UNP', 'UNS', 'UNVR', 'UONE', 'UPFC', 'UPLD', 'UPLMQ', 'UPS', 'UPWK', 'URBN', 'URGN', 'URI', 'URS', 'USAC', 'USAT', 'USB', 'USCR', 'USEG', 'USFD', 'USG', 'USM', 'UST1', 'USX', 'UTEK', 'UTHR', 'UTI', 'UTX-PA', 'UTX', 'UTZ', 'UVE', 'UVV', 'V', 'VA', 'VAC', 'VAL1', 'VALE-P', 'VALE', 'VAPO', 'VAR', 'VARI', 'VASC', 'VBTX', 'VC', 'VCBI', 'VCEL', 'VCI', 'VCRA', 'VCYT', 'VEC.W', 'VECO', 'VEDL', 'VEEV', 'VER', 'VERI', 'VERX', 'VFC', 'VG', 'VGR', 'VHC', 'VHS', 'VIA', 'VIAB', 'VIAC', 'VIAO', 'VIAV', 'VICI', 'VICR', 'VIE', 'VIMC', 'VIOT', 'VIPS', 'VIR', 'VIRT', 'VIT', 'VITC', 'VITL', 'VIV', 'VIVO', 'VKTX', 'VLCM', 'VLO', 'VLP', 'VLRS', 'VLTR', 'VLY', 'VM', 'VMAC', 'VMC', 'VMD', 'VMED', 'VMI', 'VMW', 'VNDA', 'VNE', 'VNET', 'VNO', 'VNOM', 'VOCS', 'VOD', 'VOLC', 'VOXX', 'VOYA', 'VPHM', 'VQ', 'VR', 'VRA', 'VREX', 'VRM', 'VRNS', 'VRNT', 'VRRM', 'VRS', 'VRSK', 'VRSN', 'VRT', 'VRTU', 'VRTX', 'VRUS', 'VRX1', 'VSAT', 'VSEA', 'VSH', 'VSI', 'VSLR', 'VSM', 'VSM.W', 'VST', 'VSTA', 'VSTO', 'VTAE', 'VTAL', 'VTIV', 'VTOL', 'VTR', 'VTR.W', 'VTRU', 'VTSS', 'VTTI', 'VVC', 'VVI', 'VVNT', 'VVPR', 'VVV', 'VVV.W', 'VWR', 'VXRT', 'VYGR', 'VZ', 'W', 'WAAS', 'WAB', 'WABC', 'WAC1', 'WAFD', 'WAGE', 'WAIR', 'WAL', 'WAT', 'WB', 'WBA', 'WBC', 'WBCO', 'WBD', 'WBK', 'WBMD', 'WBS', 'WBSN', 'WBT', 'WCC', 'WCG', 'WCIC', 'WCN', 'WCRX', 'WD', 'WDAY', 'WDC', 'WDR', 'WEB', 'WEC', 'WEDC', 'WELL-PI', 'WELL', 'WEN', 'WERN', 'WES', 'WES2', 'WEX', 'WFBI', 'WFC-PY', 'WFC-PZ', 'WFC', 'WFC.W', 'WFM', 'WGL', 'WGO', 'WH', 'WHD', 'WHR', 'WIBC', 'WIFI', 'WIMI', 'WIND', 'WING', 'WINN', 'WINT', 'WINVV', 'WIX', 'WK', 'WKHS', 'WLDN', 'WLH', 'WLK', 'WLTW', 'WM', 'WMAR', 'WMB', 'WMG', 'WMGI', 'WMS', 'WMS1', 'WMT', 'WMZ', 'WNC', 'WNR', 'WNRL', 'WOLF', 'WOR', 'WORK', 'WOW', 'WP', 'WPC', 'WPF', 'WPG-PG', 'WPM', 'WPP', 'WPP1', 'WPXP', 'WPZ', 'WPZ1', 'WR', 'WRB', 'WRD', 'WRE', 'WRI', 'WRK', 'WRK.W', 'WRLS1', 'WRTC', 'WSBC', 'WSC', 'WSFS', 'WSM', 'WSO', 'WSR', 'WST', 'WSTC', 'WTFC', 'WTNY', 'WTR', 'WTRG', 'WTS', 'WU', 'WUBA', 'WVE', 'WW', 'WWAV', 'WWAY', 'WWD', 'WWE', 'WWW', 'WX', 'WY-PA', 'WY', 'WYE', 'WYND', 'WYNN', 'X', 'XAIR', 'XEC', 'XEL', 'XENT', 'XERS', 'XFOR', 'XHR', 'XJT', 'XL', 'XLNX', 'XLRN', 'XLS', 'XNCR', 'XNPT', 'XOM', 'XON', 'XONE', 'XOOM', 'XOXO', 'XP', 'XPEL', 'XPER', 'XPEV', 'XPO', 'XRAY', 'XRIT', 'XRM', 'XRX', 'XRX.W', 'XTLY', 'XTO', 'XUE', 'XXIA', 'XYL', 'YDKN', 'YDLE', 'YECO', 'YELP', 'YETI', 'YEXT', 'YHOO', 'YMAB', 'YNDX', 'YOKU', 'YTEN', 'YUM', 'YUM.W', 'YUMC', 'YY', 'Z', 'ZAYO', 'ZBH', 'ZBRA', 'ZCVVV', 'ZEN', 'ZEP', 'ZG', 'ZGEN', 'ZGNX', 'ZI', 'ZIGO', 'ZION', 'ZIP', 'ZIXI', 'ZLAB', 'ZLC', 'ZLTQ', 'ZM', 'ZNGA', 'ZNT', 'ZNTL', 'ZOES', 'ZOLL', 'ZOLT', 'ZRAN', 'ZS', 'ZSPH', 'ZTO', 'ZTS', 'ZU', 'ZUMZ', 'ZUO', 'ZYME', 'ZYXI']

def test_get_ticker_data():
    # Arrange
    ticker = "AAPL"

    # Act
    df = ticker_service.get_ticker_eod_data(ticker=ticker)
    df_sorted = df.sort_values(by='date')

    # Assert
    assert (df_sorted.shape[0] > 190)
    assert (df_sorted.columns[0] == 'ticker')


def test_get_ticker_data_prepped():
    ticker = "AAPL"

    # Act
    df = ticker_service.get_ticker_eod_data(ticker=ticker)
    df_sorted = df.sort_values(by='date')

    start_date = '2020-08-01'
    end_date = '2020-08-27'

    df_dated = df_sorted[(df['date'] > start_date) & (df['date'] < end_date)]

    df_dated.groupby()

    from sklearn.model_selection import train_test_split
    train_test_split

    from sklearn.svm import SVC
    SVC

    print(f'Num in date range: {df_dated.shape[0]}')


def test_ticker_in_range():
    # Arrange
    tickers = ['AAPL']

    date_range = DateRange(from_date=date_utils.parse_std_datestring("2020-08-01"),
                           to_date=date_utils.parse_std_datestring("2020-08-30")
                           )

    # Act
    ticker_service.get_tickers_in_range(tickers=tickers, date_range=date_range)

    # Assert


def test_ticker_on_date():
    # Arrange
    date_string = "2020-07-09"
    dt: datetime = date_utils.parse_std_datestring(date_string)

    high_price = ticker_service.get_ticker_attribute_on_date(ticker="IBM", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="NVDA", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="ALXN", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="GOOGL", dt=dt)
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="ADI", dt=dt)

    # Act
    start = time.time()
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"1st Elapsed: {end - start} seconds")

    start = time.time()
    high_price = ticker_service.get_ticker_attribute_on_date(ticker="AAPL", dt=dt)
    end = time.time()

    print(f"2nd Elapsed: {end - start} seconds")

    # Assert
    assert (high_price == 393.91)


def test_get_next_high_days():
    # Arrange
    ticker = 'AAPL'

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-07")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-08")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-09")
    assert (high_price == 455.1)

    high_price, = ticker_service.get_next_trading_day_attr(ticker=ticker, date_str="2020-08-10")
    assert (high_price == 449.93)

    close, = ticker_service.get_next_trading_day_attr(ticker=ticker, equity_fields=[EquityFields.close], date_str="2020-08-10")
    assert (close == 437.5)

    close, high = ticker_service.get_next_trading_day_attr(ticker=ticker, equity_fields=[EquityFields.close, EquityFields.high], date_str="2020-08-08")
    assert (close == 450.91)
    assert (high == 455.1)


def get_test_tweets_and_stocks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tweet_rows = [
        {"f22_ticker": "AAA", "date": "2020-01-01", "close": ".11"},
        {"f22_ticker": "AAA", "date": "2020-01-02", "close": ".22"},
        {"f22_ticker": "BBB", "date": "2020-01-01", "close": ".33"},
        {"f22_ticker": "CCC", "date": "2020-01-01", "close": ".44"},
        {"f22_ticker": "CCC", "date": "2020-01-02", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    stock_rows = [
        {"ticker": "AAA", "date": "2020-01-01", "close": "1.11"},
        {"ticker": "AAA", "date": "2020-01-02", "close": "2.22"},
        {"ticker": "AAA", "date": "2020-01-03", "close": "21.02"},
        {"ticker": "BBB", "date": "2020-01-01", "close": "3.33"},
        {"ticker": "BBB", "date": "2020-01-02", "close": "31.01"},
        {"ticker": "CCC", "date": "2020-01-01", "close": "4.44"},
        {"ticker": "CCC", "date": "2020-01-02", "close": "5.55"},
        {"ticker": "CCC", "date": "2020-01-03", "close": "6.66"},
    ]
    df_stocks = pd.DataFrame(stock_rows)
    df_stocks = df_stocks.sample(frac=1.0)

    return df_tweets, df_stocks


def test_merge_future_price():
    # Arrange
    df_tweets, df_stocks = get_test_tweets_and_stocks()

    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)

    print(ttd)

    # Act

    # Assert
    assert (df_tweets.shape[0] == 5)
    assert (df_tweets.columns.all(["f22_ticker", "date", "close"]))


def test_get_equity_on_dates():
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": ".11"},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": ".22"},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": ".33"},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": ".44"},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    ttd = ticker_service.extract_ticker_tweet_dates(df_tweets)
    df = ticker_service.get_ticker_on_dates(ttd)

    print(df.head())


def test_pull_in_next_trading_day_info():
    # Arrange
    tweet_rows = [
        {"f22_ticker": "AAPL", "date": "2020-09-08", "close": ".11"},
        {"f22_ticker": "AAPL", "date": "2020-09-09", "close": ".22"},
        {"f22_ticker": "MSFT", "date": "2020-09-15", "close": ".33"},
        {"f22_ticker": "ATVI", "date": "2020-09-17", "close": ".44"},
        {"f22_ticker": "ATVI", "date": "2020-09-18", "close": ".55"},
    ]
    df_tweets = pd.DataFrame(tweet_rows)
    df_tweets = df_tweets.sample(frac=1.0)

    # Act
    df_twt_exp = ticker_service.pull_in_next_trading_day_info(df_tweets=df_tweets)

    # Assert
    df_aapl = df_twt_exp[(df_twt_exp["f22_ticker"] == "AAPL") & (df_twt_exp["date"] == "2020-09-08")]
    assert (df_aapl.shape[0] == 1)

    row_dict = dict(df_aapl.iloc[0])
    assert (row_dict["future_open"] == 117.26)
    assert (row_dict["future_low"] == 115.26)
    assert (row_dict["future_high"] == 119.14)
    assert (row_dict["future_close"] == 117.32)


def test_get_thing():
    df_tweets, df_stocks = get_test_tweets_and_stocks()


def test_get_all_tickers():
    # Arrange
    # Act
    tickers = ticker_service.get_all_tickers()

    # Assert
    assert (len(tickers) > 1000)


def test_get_tickers_w_filter():
    # Arrange
    # Act
    tickers = ticker_service.get_tickers_w_filters()

    # Assert
    print(len(tickers))
    print(tickers)
