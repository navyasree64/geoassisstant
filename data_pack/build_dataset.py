"""Builds the synthetic East India dataset (documents, GeoPackage, evaluation files).
Real data: school polygons from schools.gpkg (OSM). Everything else is synthetic (see DATA_README.md)."""
import re, json, os, random, sys
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box
from shapely.ops import nearest_points

SRC = sys.argv[1] if len(sys.argv) > 1 else "schools.gpkg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data_pack"
random.seed(42)
for d in ("data/docs", "eval", "geoserver"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

# ---------------- schools ----------------
g = gpd.read_file(SRC).cx[88.2:88.5, 22.45:22.70].copy().reset_index(drop=True)
g["centroid"] = g.to_crs(32645).geometry.centroid.to_crs(4326)
CURATED = {
 # name: (board, gender)
 "Delhi Public School": ("CBSE","co"), "Don Bosco School": ("CBSE","boys"), "Heritage Academy": ("CBSE","co"),
 "Narain Dass Bangur Memorial Multipurpose School": ("CBSE","co"), "Nava Nalanda Group of Schools": ("CBSE","co"),
 "Saint Xavier's Collegiate School": ("ICSE","boys"), "St. Jude's High School": ("ICSE","co"), "Holy Cross School": ("ICSE","co"),
 "The Calcutta Emmanuel School": ("ICSE","co"), "Auxilium Convent School": ("ICSE","girls"), "Harrow Hall": ("ICSE","co"),
 "Hare School": ("WBBSE","boys"), "Tantia High School": ("WBBSE","boys"), "B.K.Paul's Institution": ("WBBSE","boys"),
 "Baranagar Ramakrishna Mission Ashrama High School": ("WBBSE","boys"), "Chittaranjan Boys High School": ("WBBSE","boys"),
 "Dinabandhu Institution Main": ("WBBSE","boys"), "Dum Dum Baidyanath Institution": ("WBBSE","boys"),
 "Dum Dum Krishna Kumar Hindu Academy": ("WBBSE","boys"), "Pandit Motilal Institute": ("WBBSE","boys"),
 "Sri Ramkrishna Vidyasram": ("WBBSE","boys"), "Vivekananda School": ("WBBSE","boys"),
 "Bethune Collegiate School": ("WBBSE","girls"), "Chittaranjan Girls High School": ("WBBSE","girls"),
 "Hindu Girl's School": ("WBBSE","girls"), "Kamala Chatterjee Girls School": ("WBBSE","girls"),
 "Debiswari Girl's High School": ("WBBSE","girls"), "Rajarhat Shiksha Niketan High School for Girls": ("WBBSE","girls"),
 "Rajarhat Shiksha Niketan High School for Boys": ("WBBSE","boys"), "Kabitirtha Vidyalaya": ("WBBSE","boys"),
}
SEATS = {  # latest approved seats
 "Delhi Public School":160,"Don Bosco School":120,"Heritage Academy":100,"Narain Dass Bangur Memorial Multipurpose School":100,"Nava Nalanda Group of Schools":90,
 "Saint Xavier's Collegiate School":90,"St. Jude's High School":70,"Holy Cross School":80,"The Calcutta Emmanuel School":60,"Auxilium Convent School":50,"Harrow Hall":40,
 "Hare School":120,"Tantia High School":90,"B.K.Paul's Institution":100,"Baranagar Ramakrishna Mission Ashrama High School":80,"Chittaranjan Boys High School":60,
 "Dinabandhu Institution Main":70,"Dum Dum Baidyanath Institution":110,"Vivekananda School":60,
 "Bethune Collegiate School":100,"Chittaranjan Girls High School":80,"Hindu Girl's School":90,"Kamala Chatterjee Girls School":70,"Debiswari Girl's High School":60,
 "Rajarhat Shiksha Niketan High School for Girls":70,
 "Dum Dum Krishna Kumar Hindu Academy":80,"Pandit Motilal Institute":70,"Sri Ramkrishna Vidyasram":50,"Kabitirtha Vidyalaya":60,"Rajarhat Shiksha Niketan High School for Boys":70,
}
OLD_CBSE = {"Delhi Public School":150,"Don Bosco School":120,"Heritage Academy":90,"Narain Dass Bangur Memorial Multipurpose School":100,"Nava Nalanda Group of Schools":80}
CUTOFF = {
 "Hare School":78,"Tantia High School":72,"B.K.Paul's Institution":70,"Baranagar Ramakrishna Mission Ashrama High School":80,"Chittaranjan Boys High School":65,
 "Dinabandhu Institution Main":62,"Dum Dum Baidyanath Institution":68,"Vivekananda School":60,
 "Bethune Collegiate School":82,"Chittaranjan Girls High School":70,"Hindu Girl's School":75,"Kamala Chatterjee Girls School":68,"Debiswari Girl's High School":60,
 "Rajarhat Shiksha Niketan High School for Girls":66,
 "Dum Dum Krishna Kumar Hindu Academy":64,"Pandit Motilal Institute":61,"Sri Ramkrishna Vidyasram":58,"Kabitirtha Vidyalaya":55,"Rajarhat Shiksha Niketan High School for Boys":66,
}
LOCALITIES = {"Howrah":(88.31,22.59),"Shibpur":(88.30,22.56),"Salkia":(88.34,22.60),"Bally":(88.34,22.65),"Santragachi":(88.28,22.58),
 "Park Street":(88.36,22.55),"Ballygunge":(88.365,22.525),"Tollygunge":(88.35,22.50),"Behala":(88.31,22.50),"Jadavpur":(88.37,22.50),
 "Salt Lake":(88.41,22.58),"Dum Dum":(88.40,22.62),"Baranagar":(88.37,22.64),"Belgachia":(88.38,22.60),"Entally":(88.38,22.56)}
def loc_of(pt):
    return min(LOCALITIES, key=lambda k: (LOCALITIES[k][0]-pt.x)**2+(LOCALITIES[k][1]-pt.y)**2)
g["locality"] = g.centroid.apply(loc_of)
g["board"]=None; g["seats"]=None; g["cutoff_marks"]=None
missing=[n for n in CURATED if n not in set(g.name.dropna())]
assert not missing, missing
for i,r in g.iterrows():
    if r["name"] in CURATED and g.loc[:i,"name"].eq(r["name"]).sum()==1:
        g.at[i,"board"]=CURATED[r["name"]][0]; g.at[i,"seats"]=SEATS[r["name"]]
        if r["name"] in CUTOFF: g.at[i,"cutoff_marks"]=CUTOFF[r["name"]]
cur = g[g.board.notna()].copy()
cinfo = {r["name"]: dict(osm_id=r["osm_id"], locality=r["locality"], board=r["board"], seats=int(r["seats"]),
         cutoff=(None if r["cutoff_marks"] is None or r["cutoff_marks"]!=r["cutoff_marks"] else int(r["cutoff_marks"])),
         lon=r["centroid"].x, lat=r["centroid"].y) for _,r in cur.iterrows()}

# ---------------- roads (synthetic polylines through real school centroids) ----------------
P = cur.copy(); P["cp"]=P.centroid
gp = gpd.GeoSeries(P.cp, crs=4326).to_crs(32645)
names=list(P.name); pts=list(gp)
seeds=[0]
while len(seeds)<5:
    seeds.append(max(range(len(pts)), key=lambda i: min(pts[i].distance(pts[s]) for s in seeds)))
used=set(); ROADS={}
rnames=["Barrackpore Trunk Road","AJC Bose Road","Grand Trunk Road","Diamond Harbour Road","Belgachia Road"]
for rn,s in zip(rnames,seeds):
    cand=sorted([i for i in range(len(pts)) if i not in used], key=lambda i: pts[i].distance(pts[s]))[:5]
    used.update(cand); ROADS[rn]=[names[i] for i in cand]
roads_rows=[]
for k,(rn,sch) in enumerate(ROADS.items()):
    ps=[gp.iloc[names.index(n)] for n in sch]
    # order by projection on principal axis
    xs=[p.x for p in ps]; ys=[p.y for p in ps]
    ax=(max(xs)-min(xs) >= max(ys)-min(ys))
    ps=sorted(ps,key=lambda p:(p.x if ax else p.y))
    roads_rows.append(dict(osm_id=f"R{k+1:03d}", name=rn, fclass="primary", geometry=LineString(ps)))
roads = gpd.GeoDataFrame(roads_rows, crs=32645).to_crs(4326)
# ---------------- landuse (synthetic rectangles) ----------------
classes=["residential","commercial","industrial","park","residential","commercial","residential","park","residential","commercial","residential","industrial","residential","commercial","residential"]
lu=[]
for (ln,(x,y)),c in zip(LOCALITIES.items(),classes):
    lu.append(dict(osm_id=f"L{len(lu)+1:03d}", name=ln, fclass=c, geometry=box(x-0.008,y-0.007,x+0.008,y+0.007)))
landuse=gpd.GeoDataFrame(lu,crs=4326)
gout=g.drop(columns=["centroid"])
gpk=f"{OUT}/data/east_india.gpkg"
if os.path.exists(gpk): os.remove(gpk)
gout.to_file(gpk,layer="schools",driver="GPKG")
roads.to_file(gpk,layer="roads",driver="GPKG")
landuse.to_file(gpk,layer="landuse",driver="GPKG")

# near relations reference (<=300 m, EPSG:32646 as the task requires)
sch46=cur.to_crs(32646); rd46=roads.to_crs(32646)
NEAR=[]
for i,r in sch46.iterrows():
    for j,rr in rd46.iterrows():
        d=r.geometry.distance(rr.geometry)
        if d<=300: NEAR.append(dict(school=r["name"],osm_id=r["osm_id"],road=rr["name"],distance_m=round(d,1)))

# ---------------- documents ----------------
TAG=re.compile(r"\{(S|B|C|R|L|Rd|D|Q):([^{}=|]*?)(?:=([^{}|]*?))?(?:\|([^{}]*?))?\}")
TYPES=dict(S="School",B="Board",C="Circular",R="Rule",L="Locality",Rd="Road",D="Date",Q="Quantity")
def parse(paras):
    text=""; ents=[]; chunks=[]
    for n,(key,mk) in enumerate(paras):
        if n: text+="\n\n"
        pstart=len(text); out=""; pos=0
        for m in TAG.finditer(mk):
            out+=mk[pos:m.start()]; s=pstart+len(out)
            surf=m.group(2); out+=surf
            ents.append(dict(type=TYPES[m.group(1)],start=s,end=s+len(surf),text=surf,canonical=(m.group(3) or surf),kind=m.group(4)))
            pos=m.end()
        out+=mk[pos:]; text+=out
        chunks.append(dict(key=key,start=pstart,end=pstart+len(out),text=out))
    for e in ents: assert text[e["start"]:e["end"]]==e["text"],(e,text[e["start"]:e["end"]])
    return text,ents,chunks
def tbl(head,rows):
    return head+"\n| School | Seats | Cut-off marks |\n|---|---|---|\n"+"\n".join(rows) if len(head) else ""
def S(n,var=None):
    return "{S:%s%s}"%(var or n, "=%s"%n if var else "")
def Q(v,txt=None,kind="seats"): return "{Q:%s=%s|%s}"%(txt or v,v,kind)
def seat_rows(names,cut=True):
    return [f"| {S(n)} | {Q(SEATS[n])} | {Q(CUTOFF[n],kind='marks')} |" if cut else f"| {S(n)} | {Q(SEATS[n])} |" for n in names]
def D(txt,iso): return "{D:%s=%s}"%(txt,iso)
BOYS=["Hare School","Tantia High School","B.K.Paul's Institution","Baranagar Ramakrishna Mission Ashrama High School","Chittaranjan Boys High School","Dinabandhu Institution Main","Dum Dum Baidyanath Institution","Vivekananda School"]
GIRLS=["Bethune Collegiate School","Chittaranjan Girls High School","Hindu Girl's School","Kamala Chatterjee Girls School","Debiswari Girl's High School","Rajarhat Shiksha Niketan High School for Girls"]
OTHER=["Dum Dum Krishna Kumar Hindu Academy","Pandit Motilal Institute","Sri Ramkrishna Vidyasram","Kabitirtha Vidyalaya","Rajarhat Shiksha Niketan High School for Boys"]
CBSE=["Delhi Public School","Don Bosco School","Heritage Academy","Narain Dass Bangur Memorial Multipurpose School","Nava Nalanda Group of Schools"]
ICSE=["Saint Xavier's Collegiate School","St. Jude's High School","Holy Cross School","The Calcutta Emmanuel School","Auxilium Convent School","Harrow Hall"]
def cnum(n): return f"DEO/ADM/2026/{n:03d}"
def hdr(n,date_txt,iso,subject): return f"# Circular No. {{C:{cnum(n)}}}\nDate: {D(date_txt,iso)}\nSubject: {subject}"
A_ALL="{B:CBSE}, {B:ICSE} and {B:WBBSE}"
def seat_tab(head,names,seats): return head+"\n| School | Seats |\n|---|---|\n"+"\n".join(f"| {S(n)} | {Q(seats[n])} |" for n in names)

DOCS=[]  # (doc_id, filename, meta, paras)
def add(did,fn,meta,paras): DOCS.append((did,fn,meta,paras))
add("CIR-001","CIR-001.md",dict(number=cnum(1),date="2026-01-10",applies=["WBBSE"],supersedes=None),[
 ("head",hdr(1,"10 January 2026","2026-01-10","Admission schedule for {B:WBBSE} schools in the district")),
 ("sched","This circular applies to all schools affiliated with the {B:West Bengal Board of Secondary Education=WBBSE} ({B:WBBSE}). Application forms for Class V admission will be available from "+D("1 February 2026","2026-02-01")+" and the last date for submission is "+D("28 February 2026","2026-02-28")+". The admission lottery will be held on "+D("15 March 2026","2026-03-15")+"."),
 ("age","A child must be at least "+Q(9,"9 years","age_years")+" old on "+D("1 January 2026","2026-01-01")+" to apply for Class V. The age limit for Class I admission is "+Q(5,"5 years","age_years")+". Schools such as "+S("Hare School")+", "+S("Bethune Collegiate School")+" and "+S("Tantia High School")+" must display the schedule on their notice boards."),
 ("contact","Parents may submit queries to the District Education Office in {L:Howrah} or {L:Salkia}. Under {R:Rule 3} of the district admission guidelines, no school may charge a fee for the application form.")])
add("CIR-002","CIR-002.md",dict(number=cnum(2),date="2026-01-15",applies=["CBSE"],supersedes=None),[
 ("head",hdr(2,"15 January 2026","2026-01-15","Seat matrix and age limit for {B:CBSE} schools")),
 ("body","This circular applies to all schools affiliated with the {B:Central Board of Secondary Education=CBSE} ({B:CBSE}). The minimum age for Class I admission is "+Q(6,"6 years","age_years")+" as on "+D("31 March 2026","2026-03-31")+". Admissions follow the lottery described in the district lottery guidelines."),
 ("table",seat_tab("Approved seats for Class I:",CBSE,OLD_CBSE)),
 ("note",S("Heritage Academy")+" and "+S("Nava Nalanda Group of Schools")+" must complete registration by "+D("31 January 2026","2026-01-31")+".")])
add("CIR-003","CIR-003.md",dict(number=cnum(3),date="2026-01-20",applies=["ICSE"],supersedes=None),[
 ("head",hdr(3,"20 January 2026","2026-01-20","Seat matrix and age limit for {B:ICSE} schools")),
 ("body","This circular applies to all schools affiliated with the {B:Council for the Indian School Certificate Examinations=ICSE} ({B:ICSE}). The minimum age for Class I admission is "+Q(5,"5 years","age_years")+". Admissions follow the lottery described in the district lottery guidelines."),
 ("table",seat_tab("Approved seats for Class I:",ICSE,SEATS)),
 ("note","The new {B:ICSE} school {S:Sunrise International School} in {L:Salt Lake} has been allotted "+Q(55,"55 seats")+". "+S("Saint Xavier's Collegiate School","St Xavier's Collegiate School")+" will publish its registration list by "+D("10 February 2026","2026-02-10")+".")])
add("CIR-004","CIR-004.md",dict(number=cnum(4),date="2026-02-05",applies=["WBBSE"],supersedes=cnum(1)),[
 ("head",hdr(4,"5 February 2026","2026-02-05","Revised admission schedule for {B:WBBSE} schools")),
 ("supers","This circular supersedes Circular {C:"+cnum(1)+"} dated "+D("10 January 2026","2026-01-10")+"."),
 ("rev","For {B:WBBSE} schools the last date for Class V application is extended to "+D("10 March 2026","2026-03-10")+" and the admission lottery will be held on "+D("25 March 2026","2026-03-25")+". All other conditions, including the age limit of "+Q(9,"9 years","age_years")+", remain unchanged.")])
add("CIR-005","CIR-005.md",dict(number=cnum(5),date="2026-02-12",applies=["CBSE","ICSE","WBBSE"],supersedes=None),[
 ("head",hdr(5,"12 February 2026","2026-02-12","Reservation for economically weaker sections (EWS)")),
 ("rule4","This circular applies to all schools affiliated with "+A_ALL+". Under {R:Rule 4(1)}, "+Q(25,"25 percent","percent")+" of seats in Class I must be reserved for children from economically weaker sections (EWS)."),
 ("rule5","Under {R:Rule 5}, an applicant must submit an income certificate showing annual family income below "+Q(120000,"Rs. 120000","rupees")+". Schools such as "+S("Delhi Public School")+" and "+S("Don Bosco School")+" must publish the number of EWS seats on their websites.")])
add("CIR-006","CIR-006.md",dict(number=cnum(6),date="2026-02-20",applies=["WBBSE"],supersedes=None),[
 ("head",hdr(6,"20 February 2026","2026-02-20","Class V seats and cut-off marks for {B:WBBSE} boys' schools")),
 ("intro","This circular applies to {B:WBBSE} boys' schools. It lists the seats for Class V and the cut-off marks in the admission test (out of 100)."),
 ("table",tbl("Seats and cut-off marks:",seat_rows(BOYS))),
 ("note",S("Baranagar Ramakrishna Mission Ashrama High School")+" has the highest cut-off ("+Q(80,"80 marks","marks")+") and "+S("Hare School")+" has the most seats ("+Q(120,"120 seats")+").")])
add("CIR-007","CIR-007.md",dict(number=cnum(7),date="2026-02-22",applies=["WBBSE"],supersedes=None),[
 ("head",hdr(7,"22 February 2026","2026-02-22","Class V seats and cut-off marks for {B:WBBSE} girls' schools")),
 ("intro","This circular applies to {B:WBBSE} girls' schools. It lists the seats for Class V and the cut-off marks in the admission test (out of 100)."),
 ("table",tbl("Seats and cut-off marks:",seat_rows(GIRLS))),
 ("note",S("Bethune Collegiate School")+" has the highest cut-off ("+Q(82,"82 marks","marks")+") and the most seats ("+Q(100,"100 seats")+").")])
add("CIR-008","CIR-008.md",dict(number=cnum(8),date="2026-03-01",applies=["CBSE","ICSE","WBBSE"],supersedes=None),[
 ("head",hdr(8,"1 March 2026","2026-03-01","Mid-session transfer admissions")),
 ("rule7","Under {R:Rule 7}, a student seeking transfer must produce a transfer certificate issued by the previous school. Vacancies must be filled within "+Q(30,"30 days","days")+" of the notice."),
 ("applies","This circular applies to schools affiliated with "+A_ALL+". "+S("Dum Dum Krishna Kumar Hindu Academy")+" and "+S("Pandit Motilal Institute")+" have reported vacancies.")])
road_paras=[]
for rn,sch in ROADS.items():
    slug=re.sub(r"\W+","_",rn.lower()).strip("_")
    road_paras.append(("r_"+slug,"Schools on {Rd:%s}: %s."%(rn,", ".join(S(n) for n in sch))))
add("CIR-009","CIR-009.md",dict(number=cnum(9),date="2026-03-05",applies=["CBSE","ICSE","WBBSE"],supersedes=None),[
 ("head",hdr(9,"5 March 2026","2026-03-05","Road safety near schools")),
 ("intro","This circular applies to schools affiliated with "+A_ALL+". Schools located on the following roads must install speed breakers and zebra crossings within "+Q(45,"45 days","days")+".")]+road_paras)
add("CIR-010","CIR-010.md",dict(number=cnum(10),date="2026-03-10",applies=["WBBSE"],supersedes=None),[
 ("head",hdr(10,"10 March 2026","2026-03-10","Class V seats and cut-off marks for other {B:WBBSE} schools")),
 ("intro","This circular applies to {B:WBBSE} schools not covered by earlier seat circulars. The cut-off marks are out of 100."),
 ("table",tbl("Seats and cut-off marks:",seat_rows(OTHER)))])
add("CIR-011","CIR-011.md",dict(number=cnum(11),date="2026-03-18",applies=["CBSE","ICSE","WBBSE"],supersedes=cnum(5)),[
 ("head",hdr(11,"18 March 2026","2026-03-18","Revised income ceiling for EWS applicants")),
 ("supers","This circular supersedes Circular {C:"+cnum(5)+"} and applies to schools affiliated with "+A_ALL+"."),
 ("rule5","Under {R:Rule 5} (as amended), the annual family income ceiling is revised to "+Q(150000,"Rs. 150000","rupees")+"."),
 ("rule4","{R:Rule 4(1)} is unchanged: "+Q(25,"25 percent","percent")+" of Class I seats remain reserved for EWS children.")])
NEWCB={**{n:SEATS[n] for n in CBSE},"Lakeview Public School":70}
add("CIR-012","CIR-012.md",dict(number=cnum(12),date="2026-04-02",applies=["CBSE"],supersedes=cnum(2)),[
 ("head",hdr(12,"2 April 2026","2026-04-02","Revised seat matrix for {B:CBSE} schools")),
 ("supers","This circular supersedes Circular {C:"+cnum(2)+"} dated "+D("15 January 2026","2026-01-15")+" and applies to {B:CBSE} schools."),
 ("table",seat_tab("Revised seats for Class I:",CBSE+["Lakeview Public School"],NEWCB).replace("| Lakeview Public School |","| {S:Lakeview Public School} |").replace("| {S:Lakeview Public School} | 70 |","| {S:Lakeview Public School} | "+Q(70)+" |")),
 ("note","The new {B:CBSE} school {S:Lakeview Public School} has been allotted "+Q(70,"70 seats")+". "+S("Delhi Public School")+" has the most seats ("+Q(160,"160 seats")+").")])
add("GUIDE-001","GUIDE-001.txt",dict(number=None,date="2026-01-05",applies=["CBSE","ICSE","WBBSE"],supersedes=None),[
 ("head","Guidelines for admission lotteries\nIssued on "+D("5 January 2026","2026-01-05")),
 ("rule23","{R:Rule 2.3}: The lottery draw must be held in the presence of at least "+Q(3,"3 parents","persons")+" chosen by the school."),
 ("rule25","{R:Rule 2.5}: The waiting list remains valid for "+Q(60,"60 days","days")+" after the draw."),
 ("applies","These guidelines apply to schools affiliated with "+A_ALL+".")])
# geography note (computed from data)
byloc={}
for n,i in cinfo.items(): byloc.setdefault(i["locality"],[]).append(n)
loc_paras=[]
for ln,sch in sorted(byloc.items()):
    loc_paras.append(("loc_"+re.sub(r"\W+","_",ln.lower()),"In {L:%s}, the schools covered by district circulars are %s."%(ln,", ".join("%s ({B:%s})"%(S(n),cinfo[n]["board"]) for n in sch))))
lu_txt="; ".join("{L:%s} is mainly %s"%(r["name"],r["fclass"]) for _,r in landuse.iterrows() if r["name"] in byloc)
add("NOTE-001","NOTE-001.txt",dict(number=None,date="2026-04-10",applies=[],supersedes=None),[
 ("head","Notes on schools, localities and roads in the district\nCompiled on "+D("10 April 2026","2026-04-10")),
 ("intro","This note lists where the schools mentioned in district circulars are located. Locations come from the Geo-Server layers.")]+loc_paras+[
 ("roads","The main roads covered by the road safety circular are "+", ".join("{Rd:%s}"%r for r in ROADS)+"."),
 ("landuse","Land use: "+lu_txt+".")])

CH={}; ALLE={}; chunks_out=[]; DOCMETA={}
for did,fn,meta,paras in DOCS:
    text,ents,chunks=parse(paras)
    with open(f"{OUT}/data/docs/{fn}","w",encoding="utf-8",newline="\n") as f: f.write(text)
    ALLE[did]=(text,ents)
    for c in chunks:
        cid=f"{did}-{c['key']}"; CH[(did,c["key"])]=cid
        chunks_out.append(dict(chunk_id=cid,doc_id=did,file=fn,start=c["start"],end=c["end"],page=1,text=c["text"]))
    DOCMETA[did]=dict(file=fn,**meta)
def cid(d,k): return CH[(d,k)]
with open(f"{OUT}/eval/chunks_reference.jsonl","w",encoding="utf-8",newline="\n") as f:
    for c in chunks_out: f.write(json.dumps(c)+"\n")
GOLD_DOCS=["CIR-001","CIR-002","CIR-003","CIR-004","CIR-005","CIR-006","CIR-009","CIR-011","CIR-012","NOTE-001"]
with open(f"{OUT}/eval/gold_entities.jsonl","w",encoding="utf-8",newline="\n") as f:
    for d in GOLD_DOCS:
        text,ents=ALLE[d]
        f.write(json.dumps(dict(doc_id=d,file=DOCMETA[d]["file"],entities=ents))+"\n")

# reference relations (for checking your graph)
rels=[]
for d,m in DOCMETA.items():
    for b in m["applies"]: rels.append(dict(type="applies_to",source=d,target=b))
    if m["supersedes"]: rels.append(dict(type="supersedes",source=d,target=[k for k,v in DOCMETA.items() if v.get("number")==m["supersedes"]][0]))
    text,ents=ALLE[d]
    for n in sorted({e["canonical"] for e in ents if e["type"]=="School"}): rels.append(dict(type="mentions",source=d,target=n))
for n,i in cinfo.items():
    rels.append(dict(type="affiliated_with",source=n,target=i["board"])); rels.append(dict(type="located_in",source=n,target=i["locality"]))
for x in NEAR: rels.append(dict(type="near",source=x["school"],target=x["road"],distance_m=x["distance_m"]))
json.dump(rels,open(f"{OUT}/eval/relations_reference.json","w"),indent=1)

# ---------------- evaluation questions ----------------
def J(*a): return [dict(chunk_id=cid(d,k),grade=gr) for d,k,gr in a]
def hole_school():
    return max(byloc,key=lambda k:len(byloc[k]))
HL=hole_school(); HLK="loc_"+re.sub(r"\W+","_",HL.lower())
R1=list(ROADS)[2]; R1K="r_"+re.sub(r"\W+","_",R1.lower()).strip("_")
RQ=[ # qid, type, split, query, judgments
 ("Q01","exact_identifier","tune","Circular DEO/ADM/2026/004",J(("CIR-004","head",2),("CIR-004","supers",2),("CIR-004","rev",1))),
 ("Q02","exact_identifier","holdout","DEO/ADM/2026/011",J(("CIR-011","head",2),("CIR-011","supers",1),("CIR-011","rule5",1))),
 ("Q03","exact_identifier","tune","Rule 4(1)",J(("CIR-005","rule4",2),("CIR-011","rule4",2))),
 ("Q04","exact_identifier","tune","Rule 7 transfer certificate",J(("CIR-008","rule7",2))),
 ("Q05","exact_identifier","tune","DEO/ADM/2026/006 seat table",J(("CIR-006","table",2),("CIR-006","head",1),("CIR-006","intro",1))),
 ("Q06","exact_identifier","tune","Rule 2.5",J(("GUIDE-001","rule25",2))),
 ("Q07","exact_identifier","holdout","DEO/ADM/2026/012",J(("CIR-012","head",2),("CIR-012","supers",2),("CIR-012","table",1))),
 ("Q08","exact_identifier","tune","Circular DEO/ADM/2026/009",J(("CIR-009","head",2),("CIR-009","intro",1))),
 ("Q09","paraphrase","tune","How old must a child be for Class V in WBBSE schools?",J(("CIR-001","age",2),("CIR-004","rev",1))),
 ("Q10","paraphrase","tune","Minimum age for first-grade admission in CBSE schools",J(("CIR-002","body",2))),
 ("Q11","paraphrase","holdout","Until when can parents apply for fifth grade admission in West Bengal board schools?",J(("CIR-004","rev",2),("CIR-001","sched",1))),
 ("Q12","paraphrase","tune","What proportion of seats is kept for economically weaker families?",J(("CIR-005","rule4",2),("CIR-011","rule4",2))),
 ("Q13","paraphrase","holdout","What income limit is set for EWS applicants?",J(("CIR-011","rule5",2),("CIR-005","rule5",1))),
 ("Q14","paraphrase","tune","How long does the waiting list stay valid?",J(("GUIDE-001","rule25",2))),
 ("Q15","paraphrase","tune","Documents needed to transfer to another school mid-year",J(("CIR-008","rule7",2))),
 ("Q16","paraphrase","holdout","How many seats does Delhi Public School have for Class I?",J(("CIR-012","table",2),("CIR-002","table",1),("CIR-012","note",1))),
 ("Q17","paraphrase","holdout","How many marks are needed to enter Hare School?",J(("CIR-006","table",2),("CIR-006","note",1))),
 ("Q18","paraphrase","tune","Which girls' school has the highest cut-off?",J(("CIR-007","note",2),("CIR-007","table",1))),
 ("Q19","multi_hop","holdout","Which circulars apply to the board of Hare School?",J(("CIR-001","sched",2),("CIR-004","rev",2),("CIR-005","rule4",2),("CIR-006","intro",2),("CIR-008","applies",2),("CIR-009","intro",2),("CIR-010","intro",2),("CIR-011","supers",2),("GUIDE-001","applies",1))),
 ("Q20","multi_hop","tune","Which circular was replaced by DEO/ADM/2026/004?",J(("CIR-004","supers",2),("CIR-001","head",1))),
 ("Q21","multi_hop","tune","Which circular supersedes DEO/ADM/2026/002 and what are the new seats of Heritage Academy?",J(("CIR-012","supers",2),("CIR-012","table",2),("CIR-002","table",1))),
 ("Q22","multi_hop","holdout",f"Which schools lie on {R1}?",J(("CIR-009",R1K,2),("NOTE-001","roads",1))),
 ("Q23","multi_hop","tune",f"Which schools are in {HL} and which board are they under?",J(("NOTE-001",HLK,2))),
 ("Q24","multi_hop","tune","Which board does Saint Xavier's Collegiate School follow and how many seats does it have?",J(("CIR-003","table",2),("CIR-003","body",2),("CIR-003","note",1))),
 ("Q25","list_table","tune","List the WBBSE boys' schools with more than 100 seats",J(("CIR-006","table",2),("CIR-010","table",1))),
 ("Q26","list_table","holdout","ICSE schools and their approved seats",J(("CIR-003","table",2))),
 ("Q27","list_table","tune","Cut-off marks of girls' schools",J(("CIR-007","table",2))),
 ("Q28","list_table","tune","Cut-off marks for Kabitirtha Vidyalaya and Pandit Motilal Institute",J(("CIR-010","table",2))),
 ("Q29","list_table","tune","Which CBSE schools are in the revised seat matrix?",J(("CIR-012","table",2),("CIR-002","table",1))),
 ("Q30","list_table","holdout","Schools with 60 seats or fewer among WBBSE boys' schools",J(("CIR-006","table",2),("CIR-010","table",1))),
]
assert sum(1 for q in RQ if q[2]=="tune")==20 and sum(1 for q in RQ if q[2]=="holdout")==10
with open(f"{OUT}/eval/retrieval_judgments.jsonl","w",encoding="utf-8",newline="\n") as f:
    for q in RQ: f.write(json.dumps(dict(qid=q[0],type=q[1],split=q[2],query=q[3],judgments=q[4]))+"\n")

def tabkey(n):
    if n in BOYS: return "CIR-006","table"
    if n in GIRLS: return "CIR-007","table"
    return "CIR-010","table"
CQ=[]
def cq(q,j): CQ.append((q,j))
for n in ["Hare School","Tantia High School","Chittaranjan Boys High School","Dum Dum Baidyanath Institution","Vivekananda School","Hindu Girl's School","Kamala Chatterjee Girls School","Debiswari Girl's High School"]:
    d,k=tabkey(n); cq(f"What is the cut-off mark for {n}?",J((d,k,2)))
for n in ["Pandit Motilal Institute","Sri Ramkrishna Vidyasram","Rajarhat Shiksha Niketan High School for Boys"]:
    d,k=tabkey(n); cq(f"How many seats and what cut-off does {n} have?",J((d,k,2),("CIR-010","intro",1)))
for n in ["Don Bosco School","Heritage Academy","Nava Nalanda Group of Schools"]:
    cq(f"How many Class I seats are approved for {n}?",J(("CIR-012","table",2),("CIR-002","table",1)))
for n in ["St. Jude's High School","Holy Cross School","Auxilium Convent School"]:
    cq(f"How many seats does {n} have for Class I?",J(("CIR-003","table",2)))
cq("What is the last date for submitting Class V applications in WBBSE schools?",J(("CIR-004","rev",2),("CIR-001","sched",1)))
cq("When is the admission lottery for WBBSE schools?",J(("CIR-004","rev",2),("CIR-001","sched",1)))
cq("What is the minimum age for Class I in ICSE schools?",J(("CIR-003","body",2)))
cq("What percentage of seats is reserved for EWS children?",J(("CIR-005","rule4",2),("CIR-011","rule4",2)))
cq("How many parents must attend the lottery draw?",J(("GUIDE-001","rule23",2)))
cq("Within how many days must vacancies be filled after a transfer notice?",J(("CIR-008","rule7",2)))
cq(f"Which schools are located on {list(ROADS)[0]}?",J(("CIR-009","r_"+re.sub(r"\W+","_",list(ROADS)[0].lower()).strip("_"),2)))
cq(f"Which schools are located on {list(ROADS)[1]}?",J(("CIR-009","r_"+re.sub(r"\W+","_",list(ROADS)[1].lower()).strip("_"),2)))
cq("Within how many days must schools on listed roads install speed breakers?",J(("CIR-009","intro",2)))
cq(f"Which schools covered by circulars are in {HL}?",J(("NOTE-001",HLK,2)))
cq("What is the land use of the localities in the district?",J(("NOTE-001","landuse",2)))
cq("What is the age requirement for Class V admission in WBBSE schools?",J(("CIR-001","age",2),("CIR-004","rev",1)))
cq("Which circular supersedes DEO/ADM/2026/001?",J(("CIR-004","supers",2)))
cq("Which circular revised the EWS income ceiling?",J(("CIR-011","supers",2),("CIR-011","rule5",2)))
cq("Which boys' school has the highest cut-off among WBBSE boys' schools?",J(("CIR-006","note",2),("CIR-006","table",1)))
UNANS=["What is the tuition fee at Hare School?","Who is the district education officer?","What is the school bus timetable for Delhi Public School?",
 "What are the hostel rules at St. Jude's High School?","When does the monsoon holiday begin?","How many teachers does Bethune Collegiate School have?",
 "What is the weather in Howrah today?","What is the mid-day meal menu in WBBSE schools?"]
for q in UNANS: cq(q,[])
assert len(CQ)==40,len(CQ)
idx=list(range(40)); random.Random(7).shuffle(idx)
test_idx=set(idx[:15]); 
# make sure test has at least 3 unanswerable
un=[i for i in range(32,40)]; 
for i in un[:3]:
    if i not in test_idx:
        swap=next(t for t in test_idx if t<32); test_idx.remove(swap); test_idx.add(i)
with open(f"{OUT}/eval/calibration_questions.jsonl","w",encoding="utf-8",newline="\n") as f:
    for i,(q,j) in enumerate(CQ):
        f.write(json.dumps(dict(qid=f"CQ{i+1:02d}",question=q,split="test" if i in test_idx else "fit",answerable=bool(j),judgments=j))+"\n")

def loc_of_n(n): return cinfo[n]["locality"]
XN="Saint Xavier's Collegiate School"
hare_circ=["DEO/ADM/2026/001","DEO/ADM/2026/004","DEO/ADM/2026/005","DEO/ADM/2026/006","DEO/ADM/2026/008","DEO/ADM/2026/009","DEO/ADM/2026/010","DEO/ADM/2026/011"]
DEV=[
 dict(qid="D01",question="What is the last date to apply for Class V in WBBSE schools?",behavior="answer",reference_answer="10 March 2026 (extended by Circular DEO/ADM/2026/004, which supersedes DEO/ADM/2026/001).",key_facts=["2026-03-10","DEO/ADM/2026/004"],citations=[cid("CIR-004","rev")]),
 dict(qid="D02",question="How many Class I seats does Delhi Public School have after the revision?",behavior="answer",reference_answer="160 seats (Circular DEO/ADM/2026/012 supersedes DEO/ADM/2026/002).",key_facts=["160","DEO/ADM/2026/012"],citations=[cid("CIR-012","table")]),
 dict(qid="D03",question="What is the cut-off mark for Bethune Collegiate School?",behavior="answer",reference_answer="82 marks.",key_facts=["82"],citations=[cid("CIR-007","table")]),
 dict(qid="D04",question="Which circular supersedes DEO/ADM/2026/005 and what is the new income ceiling for EWS applicants?",behavior="answer",reference_answer="Circular DEO/ADM/2026/011 supersedes it; the new ceiling is Rs. 150000.",key_facts=["DEO/ADM/2026/011","150000"],citations=[cid("CIR-011","supers"),cid("CIR-011","rule5")]),
 dict(qid="D05",question="Which circulars apply to the board of Hare School?",behavior="answer",reference_answer="Hare School is a WBBSE school. Circulars for WBBSE: "+", ".join(hare_circ)+" (001 superseded by 004; 005 superseded by 011), plus the lottery guidelines.",key_facts=hare_circ,citations=[cid("CIR-001","sched"),cid("CIR-004","rev"),cid("CIR-005","rule4"),cid("CIR-006","intro"),cid("CIR-008","applies"),cid("CIR-009","intro"),cid("CIR-010","intro"),cid("CIR-011","supers")]),
 dict(qid="D06",question=f"Which schools are located on {R1}?",behavior="answer",reference_answer=", ".join(ROADS[R1])+".",key_facts=ROADS[R1],citations=[cid("CIR-009",R1K)]),
 dict(qid="D07",question="Where is Saint Xavier's Collegiate School located?",behavior="answer",reference_answer=f"In {loc_of_n(XN)} (about lon {cinfo[XN]['lon']:.4f}, lat {cinfo[XN]['lat']:.4f}), from the Geo-Server schools layer (osm_id {cinfo[XN]['osm_id']}).",key_facts=[loc_of_n(XN),cinfo[XN]['osm_id']],citations=["geoserver",cid("NOTE-001","loc_"+re.sub(r"\W+","_",loc_of_n(XN).lower()))]),
 dict(qid="D08",question="What is the minimum age for Class I admission in CBSE schools?",behavior="answer",reference_answer="6 years as on 31 March 2026.",key_facts=["6 years","2026-03-31"],citations=[cid("CIR-002","body")]),
 dict(qid="D09",question="How long is the admission waiting list valid?",behavior="answer",reference_answer="60 days after the draw (Rule 2.5).",key_facts=["60 days","Rule 2.5"],citations=[cid("GUIDE-001","rule25")]),
 dict(qid="D10",question="What is the tuition fee at Hare School?",behavior="abstain",reference_answer="I cannot answer this from the available sources.",key_facts=[],citations=[]),
]
with open(f"{OUT}/eval/dev_questions.jsonl","w",encoding="utf-8",newline="\n") as f:
    for d in DEV: f.write(json.dumps(d)+"\n")

# summary
print("schools in layer:",len(g),"curated:",len(cinfo),"docs:",len(DOCS),"chunks:",len(chunks_out))
print("roads:",{k:len(v) for k,v in ROADS.items()}); print("near pairs (<=300m):",len(NEAR))
print("localities with curated schools:",{k:len(v) for k,v in byloc.items()})
