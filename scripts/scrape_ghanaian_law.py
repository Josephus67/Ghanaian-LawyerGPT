#!/usr/bin/env python3
"""
Ghanaian Law Web Scraper

This script scrapes Ghanaian legal resources from various online sources to create
a dataset for fine-tuning the Ghanaian LawyerGPT model.

Sources:
- Ghana Constitution (1992)
- Ghana Legal Information Institute (GhanaLII)
- Other publicly available legal documents

Usage:
    python scrape_ghanaian_law.py

Output:
    - dataset/ghanaian_law_scraped.jsonl (Q&A pairs for training)
    - dataset/ghana_constitution_articles.json (Raw constitution articles)
"""

import json
import re
import time
import os
from typing import List, Dict, Tuple
from urllib.parse import urljoin

# Try to import requests and BeautifulSoup
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("Note: requests and beautifulsoup4 not installed.")
    print("Install with: pip install requests beautifulsoup4")


class GhanaianLawScraper:
    """Scraper for Ghanaian legal resources."""
    
    def __init__(self, output_dir: str = "../dataset"):
        self.output_dir = output_dir
        self.session = requests.Session() if SCRAPING_AVAILABLE else None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Constitution structure based on the 1992 Constitution of Ghana
        self.constitution_chapters = {
            1: "The Constitution",
            2: "Territories of Ghana",
            3: "Citizenship",
            4: "The Laws of Ghana",
            5: "Fundamental Human Rights and Freedoms",
            6: "The Directive Principles of State Policy",
            7: "Representation of the People",
            8: "The Executive",
            9: "The Council of State",
            10: "The Legislature",
            11: "The Judiciary",
            12: "Freedom and Independence of the Media",
            13: "Finance",
            14: "The Public Services",
            15: "The Armed Forces",
            16: "The Police Service",
            17: "Commission on Human Rights and Administrative Justice",
            18: "Regional Organization and Local Government",
            19: "Decentralization and Local Government",
            20: "Lands and Natural Resources",
            21: "National Culture",
            22: "Chieftaincy",
            23: "Public Holidays",
            24: "Code of Conduct for Public Officers",
            25: "Amendment of the Constitution",
            26: "Miscellaneous"
        }
        
    def scrape_ghana_constitution_online(self) -> List[Dict]:
        """
        Attempt to scrape the Ghana Constitution from online sources.
        """
        if not SCRAPING_AVAILABLE:
            print("Scraping libraries not available.")
            return []
            
        articles = []
        
        # Try GhanaLII - Ghana Legal Information Institute
        urls_to_try = [
            "https://ghalii.org/gh/legislation/constitution/1992",
            "https://www.constituteproject.org/constitution/Ghana_1996",
        ]
        
        for url in urls_to_try:
            try:
                print(f"Trying to scrape: {url}")
                response = self.session.get(url, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract text content
                    text_content = soup.get_text(separator='\n', strip=True)
                    if len(text_content) > 1000:
                        print(f"Successfully retrieved content from {url}")
                        return self._parse_constitution_text(text_content)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
                
        return articles
    
    def _parse_constitution_text(self, text: str) -> List[Dict]:
        """Parse constitution text into structured articles."""
        articles = []
        
        # Pattern to match articles
        article_pattern = r'Article\s+(\d+)[\.\:]?\s*([^\n]+)?\n(.*?)(?=Article\s+\d+|$)'
        matches = re.findall(article_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            article_num = match[0]
            title = match[1].strip() if match[1] else ""
            content = match[2].strip()
            
            if content:
                articles.append({
                    "article_number": article_num,
                    "title": title,
                    "content": content[:2000]  # Limit content length
                })
                
        return articles
    
    def generate_constitution_dataset(self) -> List[Dict]:
        """
        Generate a comprehensive dataset of Q&A pairs about the Ghana Constitution.
        This uses the official text of the 1992 Constitution.
        """
        qa_pairs = []
        
        # The 1992 Constitution of Ghana - Key Articles and Provisions
        constitution_data = self._get_ghana_constitution_data()
        
        for item in constitution_data:
            # Generate question-answer pairs for each article/provision
            qa_pairs.extend(self._generate_qa_from_article(item))
            
        return qa_pairs
    
    def _get_ghana_constitution_data(self) -> List[Dict]:
        """
        Returns structured data about the Ghana Constitution.
        This is based on the official 1992 Constitution of Ghana.
        """
        return [
            # Chapter 1 - The Constitution
            {
                "chapter": 1,
                "article": 1,
                "title": "Supremacy of the Constitution",
                "content": """(1) The Sovereignty of Ghana resides in the people of Ghana in whose name and for whose welfare the powers of government are to be exercised in the manner and within the limits laid down in this Constitution.
(2) This Constitution shall be the supreme law of Ghana and any other law found to be inconsistent with any provision of this Constitution shall, to the extent of the inconsistency, be void.
(3) The Constitution shall be the fundamental law of the nation and shall be enforced and observed as such."""
            },
            {
                "chapter": 1,
                "article": 2,
                "title": "Enforcement of the Constitution",
                "content": """(1) A person who alleges that -
(a) an enactment or anything contained in or done under the authority of that or any other enactment; or
(b) any act or omission of any person;
is inconsistent with, or is in contravention of a provision of this Constitution, may bring an action in the Supreme Court for a declaration to that effect.
(2) The Supreme Court shall, for the purposes of a declaration under clause (1) of this article, make such orders and give such directions as it may consider appropriate for giving effect, or enabling effect to be given, to the declaration so made."""
            },
            {
                "chapter": 1,
                "article": 3,
                "title": "Defence of the Constitution",
                "content": """(1) Parliament shall have no power to enact a law establishing a one-party state.
(2) Any activity of a person or group of persons which suppresses or seeks to suppress the lawful political activity of any other person or any class of persons, or persons generally is unlawful.
(3) Any person who -
(a) by himself or in concert with others by any violent or other unlawful means, suspends or overthrows or abrogates this Constitution or any part of it, or attempts to do any such act; or
(b) aids and abets in any manner any person referred to in paragraph (a) of this clause;
commits the offence of high treason and shall, upon conviction, be sentenced to suffer death."""
            },
            # Chapter 5 - Fundamental Human Rights and Freedoms
            {
                "chapter": 5,
                "article": 12,
                "title": "Protection of Fundamental Human Rights and Freedoms",
                "content": """(1) The fundamental human rights and freedoms enshrined in this Chapter shall be respected and upheld by the Executive, Legislature and Judiciary and all other organs of government and its agencies and, where applicable to them, by all natural and legal persons in Ghana, and shall be enforceable by the Courts as provided for in this Constitution.
(2) Every person in Ghana, whatever his race, place of origin, political opinion, colour, religion, creed or gender shall be entitled to the fundamental human rights and freedoms of the individual contained in this Chapter but subject to respect for the rights and freedoms of others and for the public interest."""
            },
            {
                "chapter": 5,
                "article": 13,
                "title": "Right to Life",
                "content": """(1) No person shall be deprived of his life intentionally except in the exercise of the execution of a sentence of a court in respect of a criminal offence under the laws of Ghana of which he has been convicted.
(2) A person shall not be held to have deprived another person of his life in contravention of clause (1) of this article if that other person dies as the result of a lawful act of war or if that other person dies as the result of the use of force to such extent as is reasonably justifiable in the particular circumstances."""
            },
            {
                "chapter": 5,
                "article": 14,
                "title": "Protection of Personal Liberty",
                "content": """(1) Every person shall be entitled to his personal liberty and no person shall be deprived of his personal liberty except in accordance with the procedure permitted by law in any of the following cases-
(a) in execution of a sentence or order of a court in respect of a criminal offence of which he has been convicted; or
(b) in execution of an order of a court punishing him for contempt of court; or
(c) for the purpose of bringing him before a court in execution of an order of a court; or
(d) in the case of a person suffering from an infectious or contagious disease, a person of unsound mind, a person addicted to drugs or alcohol or a vagrant, for the purpose of his care or treatment or the protection of the community; or
(e) for the purpose of the education or welfare of a person who has not attained the age of eighteen years; or
(f) for the purpose of preventing the unlawful entry of that person into Ghana, or of effecting the expulsion, extradition or other lawful removal of that person from Ghana or for the purpose of restricting that person while he is being lawfully conveyed through Ghana in the course of his extradition or removal from one country to another."""
            },
            {
                "chapter": 5,
                "article": 15,
                "title": "Respect for Human Dignity",
                "content": """(1) The dignity of all persons shall be inviolable.
(2) No person shall, whether or not he is arrested, restricted or detained, be subjected to -
(a) torture or other cruel, inhuman or degrading treatment or punishment;
(b) any other condition that detracts or is likely to detract from his dignity and worth as a human being."""
            },
            {
                "chapter": 5,
                "article": 16,
                "title": "Protection from Slavery and Forced Labour",
                "content": """(1) No person shall be held in slavery or servitude.
(2) No person shall be required to perform forced labour.
(3) For the purposes of this article, "forced labour" does not include-
(a) any labour required as a result of a sentence or order of a court;
(b) any labour required of a member of a disciplined force or service as his duties or, in the case of a person who has conscientious objections to service as a member of the Armed Forces of Ghana, any labour which that person is required by law to perform in place of such service;
(c) any labour required during any period when Ghana is at war or in the event of an emergency or calamity that threatens the life and well-being of the community."""
            },
            {
                "chapter": 5,
                "article": 17,
                "title": "Equality and Freedom from Discrimination",
                "content": """(1) All persons shall be equal before the law.
(2) A person shall not be discriminated against on grounds of gender, race, colour, ethnic origin, religion, creed or social or economic status.
(3) For the purposes of this article, "discriminate" means to give different treatment to different persons attributable only or mainly to their respective descriptions by gender, race, colour, ethnic origin, religion, creed, or social or economic status.
(4) Nothing in this article shall prevent Parliament from enacting laws that are reasonably necessary to provide-
(a) for the implementation of policies and programmes aimed at redressing social, economic or educational imbalance in the Ghanaian society."""
            },
            {
                "chapter": 5,
                "article": 18,
                "title": "Protection of Privacy of Home and Other Property",
                "content": """(1) Every person has the right to own property either alone or in association with others.
(2) No person shall be subjected to interference with the privacy of his home, property, correspondence or communication except in accordance with law and as may be necessary in a free and democratic society for public safety or the economic well-being of the country, for the protection of health or morals, for the prevention of disorder or crime or for the protection of the rights or freedoms of others."""
            },
            {
                "chapter": 5,
                "article": 19,
                "title": "Fair Trial",
                "content": """(1) A person charged with a criminal offence shall be given a fair hearing within a reasonable time by a court.
(2) A person charged with a criminal offence shall-
(a) in the case of an offence other than high treason or treason, be presumed to be innocent until he is proved or has pleaded guilty;
(b) be informed immediately in a language he understands, and in detail, of the nature of the offence charged;
(c) be given adequate time and facilities for the preparation of his defence;
(d) be permitted to defend himself before the court in person or by a lawyer of his choice;
(e) be afforded facilities to examine, in person or by his lawyer, the witnesses called by the prosecution before the court, and to obtain the attendance and carry out the examination of witnesses to testify on the same conditions as those applicable to witnesses called by the prosecution."""
            },
            {
                "chapter": 5,
                "article": 21,
                "title": "General Fundamental Freedoms",
                "content": """(1) All persons shall have the right to-
(a) freedom of speech and expression, which shall include freedom of the press and other media;
(b) freedom of thought, conscience and belief, which shall include academic freedom;
(c) freedom to practise any religion and to manifest such practice;
(d) freedom of assembly including freedom to take part in processions and demonstrations;
(e) freedom of association, which shall include freedom to form or join trade unions or other associations, national and international, for the protection of their interest;
(f) information, subject to such qualifications and laws as are necessary in a democratic society;
(g) freedom of movement which shall include freedom to move freely throughout Ghana, the right to leave and to enter Ghana and immunity from expulsion from Ghana."""
            },
            # Chapter 8 - The Executive
            {
                "chapter": 8,
                "article": 57,
                "title": "The President of Ghana",
                "content": """(1) There shall be a President of the Republic of Ghana who shall be the Head of State and Head of Government and Commander-in-Chief of the Armed Forces of Ghana.
(2) The President shall take precedence over all other persons in Ghana; and in descending order, the Vice-President, the Speaker of Parliament and the Chief Justice, shall take precedence over all other persons in Ghana."""
            },
            {
                "chapter": 8,
                "article": 58,
                "title": "Executive Authority of Ghana",
                "content": """(1) The executive authority of Ghana shall vest in the President and shall be exercised in accordance with the provisions of this Constitution.
(2) The executive authority of Ghana shall extend to the execution and maintenance of this Constitution and all laws made under or continued in force by this Constitution.
(3) Subject to the provisions of this Constitution, the functions conferred on the President by clause (1) of this article may be exercised by him either directly or through officers subordinate to him."""
            },
            {
                "chapter": 8,
                "article": 60,
                "title": "Qualification of the President",
                "content": """(1) A person shall not be qualified for election as President of Ghana unless-
(a) he is a citizen of Ghana by birth;
(b) he has attained the age of forty years; and
(c) he is a person who is otherwise qualified to be elected a member of Parliament, except that the disqualifications set out in paragraphs (c), (d) and (e) of clause (2) of article 94 of this Constitution shall not apply to him."""
            },
            {
                "chapter": 8,
                "article": 63,
                "title": "Election of President",
                "content": """(1) The election of the President shall be by universal adult suffrage and shall be by secret ballot.
(2) A candidate shall not be declared elected as President unless-
(a) he has obtained more than fifty percent of the total number of valid votes cast at the election; and
(b) the votes cast in his favour were obtained from not less than two-thirds of the regions of Ghana.
(3) Where at an election under this article there are more than two candidates and no candidate obtains the number of votes and percentages specified in clause (2) of this article, a second election shall be held within twenty-one days after the previous election."""
            },
            # Chapter 10 - The Legislature
            {
                "chapter": 10,
                "article": 93,
                "title": "Parliament of Ghana",
                "content": """(1) There shall be a Parliament of Ghana which shall consist of not less than one hundred and forty elected members.
(2) Subject to the provisions of this Constitution, the legislative power of Ghana shall be vested in Parliament and shall be exercised in accordance with this Constitution."""
            },
            {
                "chapter": 10,
                "article": 94,
                "title": "Qualification and Disqualification for Membership of Parliament",
                "content": """(1) Subject to the provisions of this article, a person shall not be qualified to be a member of Parliament unless-
(a) he is a citizen of Ghana, has attained the age of twenty-one years and is a registered voter;
(b) he is resident in the constituency for which he stands as a candidate for election to Parliament or has resided there for a total period of not less than five years out of the ten years immediately preceding the election for which he offers himself as a candidate or he hails from that constituency; and
(c) he has paid all his taxes or made arrangements satisfactory to the appropriate authority for the payment of his taxes."""
            },
            # Chapter 11 - The Judiciary
            {
                "chapter": 11,
                "article": 125,
                "title": "The Judiciary",
                "content": """(1) Justice emanates from the people and shall be administered in the name of the Republic by the Judiciary which shall be independent and subject only to this Constitution.
(2) Citizens may exercise popular participation in the administration of justice through the institution of public tribunals."""
            },
            {
                "chapter": 11,
                "article": 126,
                "title": "The Superior Courts of Judicature",
                "content": """(1) The Superior Courts of Judicature shall comprise-
(a) the Supreme Court of Ghana;
(b) the Court of Appeal; and
(c) the High Court and Regional Tribunals.
(2) Subject to the provisions of this Constitution, the Superior Courts shall be the final authority in all matters of law in Ghana, including matters relating to this Constitution."""
            },
            {
                "chapter": 11,
                "article": 127,
                "title": "Independence of the Judiciary",
                "content": """(1) In the exercise of the judicial power of Ghana, the Judiciary, in both its judicial and administrative functions, including financial administration, shall not be subject to the control or direction of any person or authority.
(2) Neither the President nor Parliament nor any person whatsoever shall interfere with Judges or judicial officers or other persons exercising judicial power, in the exercise of their judicial functions."""
            },
            # Chapter 17 - CHRAJ
            {
                "chapter": 17,
                "article": 216,
                "title": "Commission on Human Rights and Administrative Justice",
                "content": """There shall be established by Act of Parliament within six months after Parliament first meets after the coming into force of this Constitution, a Commission on Human Rights and Administrative Justice which shall consist of-
(a) a Commissioner for Human Rights and Administrative Justice; and
(b) two Deputy Commissioners for Human Rights and Administrative Justice."""
            },
            {
                "chapter": 17,
                "article": 218,
                "title": "Functions of the Commission",
                "content": """The functions of the Commission shall be defined and prescribed by Act of Parliament and shall include the duty-
(a) to investigate complaints of violations of fundamental rights and freedoms, injustice, corruption, abuse of power and unfair treatment of any person by a public officer in the exercise of his official duties;
(b) to investigate complaints concerning the functioning of the Public Services Commission, the administrative organs of the State, the Armed Forces, the Police Service and the Prisons Service in so far as the complaints relate to the failure to achieve a balanced structuring of those services or equal access by all to the recruitment of those services or fair administration in relation to those services;
(c) to investigate complaints concerning practices and actions by persons, private enterprises and other institutions where those complaints allege violations of fundamental rights and freedoms under this Constitution."""
            },
            # Chapter 22 - Chieftaincy
            {
                "chapter": 22,
                "article": 270,
                "title": "Institution of Chieftaincy",
                "content": """(1) The institution of chieftaincy, together with its traditional councils as established by customary law and usage, is hereby guaranteed.
(2) Parliament shall have no power to enact any law which-
(a) confers on any person or authority the right to accord or withdraw recognition to or from a chief for any purpose whatsoever; or
(b) in any way detracts or derogates from the honour and dignity of the institution of chieftaincy.
(3) Nothing in or done under the authority of any law shall be held to be inconsistent with, or in contravention of, clause (1) or (2) of this article if the law makes provision for-
(a) the determination, in accordance with the appropriate customary law and usage, by a traditional council, a Regional House of Chiefs or a Chieftaincy Committee of any of them, of the validity of the nomination, election, selection, installation or deposition of a person as a chief."""
            },
            {
                "chapter": 22,
                "article": 271,
                "title": "National House of Chiefs",
                "content": """(1) There shall be a National House of Chiefs.
(2) The National House of Chiefs shall consist of five paramount chiefs from each region, elected by the Regional House of Chiefs from among themselves.
(3) The National House of Chiefs shall-
(a) advise any person or authority charged with any responsibility under this Constitution or any other law for any matter relating to or affecting chieftaincy;
(b) undertake the progressive study, interpretation and codification of customary law with a view to evolving, in appropriate cases, a unified system of rules of customary law, and compiling the customary laws and lines of succession applicable to each stool or skin."""
            },
        ]
    
    def _generate_qa_from_article(self, article_data: Dict) -> List[Dict]:
        """Generate Q&A pairs from an article."""
        qa_pairs = []
        
        chapter = article_data["chapter"]
        article_num = article_data["article"]
        title = article_data["title"]
        content = article_data["content"]
        
        # Generate various types of questions
        
        # Question about what the article says
        qa_pairs.append({
            "question": f"What does Article {article_num} of the 1992 Constitution of Ghana say about {title.lower()}?",
            "answer": f"Article {article_num} of the 1992 Constitution of Ghana, titled '{title}', provides that: {content}"
        })
        
        # Question about the provision
        qa_pairs.append({
            "question": f"Explain the provisions of Article {article_num} ({title}) in the Ghanaian Constitution.",
            "answer": f"Article {article_num} of the 1992 Constitution of Ghana addresses {title}. The article states: {content}"
        })
        
        # Chapter-based question
        qa_pairs.append({
            "question": f"Under which chapter of the Ghana Constitution is {title} addressed and what are its provisions?",
            "answer": f"{title} is addressed under Chapter {chapter} of the 1992 Constitution of Ghana, specifically in Article {article_num}. The provision states: {content}"
        })
        
        return qa_pairs
    
    def generate_ghanaian_laws_dataset(self) -> List[Dict]:
        """
        Generate a dataset of Q&A pairs about major Ghanaian laws and legal concepts.
        """
        qa_pairs = []
        
        # Criminal Offences Act, 1960 (Act 29)
        criminal_law = [
            {
                "question": "What is the Criminal Offences Act, 1960 (Act 29) of Ghana?",
                "answer": "The Criminal Offences Act, 1960 (Act 29) is Ghana's primary criminal legislation that codifies criminal offences and their punishments. It defines various crimes including offences against the state, offences against the person (such as murder, manslaughter, and assault), sexual offences, offences against property (such as theft and robbery), and other miscellaneous offences. The Act has been amended several times to address contemporary criminal issues in Ghana."
            },
            {
                "question": "How is murder defined under the Criminal Offences Act of Ghana?",
                "answer": "Under Section 46 of the Criminal Offences Act, 1960 (Act 29), murder is defined as the unlawful killing of another person with the intention to cause death, or with the intention to cause grievous bodily harm, or with knowledge that the act or omission causing death will probably cause the death of or grievous bodily harm to some person. Murder is punishable by death in Ghana, although there has been a de facto moratorium on executions."
            },
            {
                "question": "What is the punishment for theft under Ghanaian criminal law?",
                "answer": "Under Section 124 of the Criminal Offences Act, 1960 (Act 29), theft is a second-degree felony and is punishable by imprisonment for a term not exceeding ten years. However, if the theft involves property of a value exceeding a specified amount, or involves certain aggravating factors, the punishment may be more severe. The Act defines theft as the fraudulent taking of a thing capable of being stolen, or fraudulently converting to use of any person other than the owner, anything capable of being stolen."
            },
            {
                "question": "What constitutes robbery under the Criminal Offences Act of Ghana?",
                "answer": "Under Section 149 of the Criminal Offences Act, 1960 (Act 29), robbery is committed when a person steals and immediately before or at the time of stealing or immediately after stealing, uses or threatens to use force on any person or property in order to obtain or retain the thing stolen or to prevent resistance to its being stolen or retained. Robbery is a first-degree felony punishable by imprisonment for a term of not less than ten years."
            },
            {
                "question": "What are the provisions for sexual offences under the Criminal Offences Act of Ghana?",
                "answer": "The Criminal Offences Act, 1960 (Act 29) addresses sexual offences in Sections 97-106. Rape is defined under Section 97 as the carnal knowledge of a female of sixteen years or above without her consent, punishable as a first-degree felony. Defilement of a child under sixteen years is addressed in Section 101, also punishable as a first-degree felony. The Act also covers incest, indecent assault, and unnatural carnal knowledge."
            },
        ]
        qa_pairs.extend(criminal_law)
        
        # Labour Act, 2003 (Act 651)
        labour_law = [
            {
                "question": "What is the Labour Act, 2003 (Act 651) of Ghana?",
                "answer": "The Labour Act, 2003 (Act 651) is Ghana's primary employment legislation that governs employment relationships in the country. It covers employment contracts, wages and salaries, hours of work and rest periods, leave entitlements, occupational health and safety, unfair termination, trade unions and collective bargaining. The Act establishes the National Labour Commission for dispute resolution and the National Tripartite Committee for labor matters."
            },
            {
                "question": "What are the working hours provisions under the Labour Act of Ghana?",
                "answer": "Under the Labour Act, 2003 (Act 651), the normal working hours for an employee shall not exceed eight hours a day or forty hours a week. An employee who works more than eight hours a day or forty hours a week is entitled to overtime pay. Night work (between 10 PM and 6 AM) attracts additional allowances. The Act also provides for rest periods of at least one hour during the working day and at least 48 consecutive hours of rest each week."
            },
            {
                "question": "What are the leave entitlements under the Labour Act of Ghana?",
                "answer": "Under the Labour Act, 2003 (Act 651), an employee is entitled to at least fifteen working days of annual leave with full pay after every twelve months of continuous service. Female employees are entitled to at least twelve weeks of maternity leave with full pay. Employees are also entitled to sick leave of their accumulated annual leave, or where the employee has no accumulated leave, sick leave of up to five days in any year."
            },
            {
                "question": "What is unfair termination under the Labour Act of Ghana?",
                "answer": "Under Section 63 of the Labour Act, 2003 (Act 651), a termination of employment is unfair if the employer fails to prove that: (a) the reason for the termination is fair; and (b) the termination was made in accordance with a fair procedure. A reason is unfair if it is based on the worker's pregnancy, union membership, race, religion, gender, or filing a complaint against the employer. The National Labour Commission can order reinstatement or compensation for unfair termination."
            },
            {
                "question": "What are the provisions for collective bargaining under the Labour Act of Ghana?",
                "answer": "The Labour Act, 2003 (Act 651) provides for collective bargaining in Sections 96-119. Workers have the right to form and join trade unions. Employers must recognize and negotiate with registered trade unions. Collective agreements are binding on the parties and their successors. The Act establishes procedures for resolving disputes arising from collective bargaining, including mediation and arbitration through the National Labour Commission."
            },
        ]
        qa_pairs.extend(labour_law)
        
        # Land Act, 2020 (Act 1036)
        land_law = [
            {
                "question": "What is the Land Act, 2020 (Act 1036) of Ghana?",
                "answer": "The Land Act, 2020 (Act 1036) is Ghana's comprehensive land legislation that consolidates the law relating to land, including the law relating to interest in land, responsibilities of owners of interest in land, and the management and administration of stool and clan lands. The Act aims to harmonize the management and administration of public and private lands and provide for related matters."
            },
            {
                "question": "What are the different types of land interests recognized under the Land Act of Ghana?",
                "answer": "Under the Land Act, 2020 (Act 1036), recognized interests in land include: (a) allodial interest - the highest interest in land; (b) usufructuary interest - the interest of a subject of a stool, skin, family or clan in the stool, skin, family or clan land; (c) leasehold interest - an interest granted by the holder of an allodial or usufructuary interest for a definite period; (d) customary freehold - an interest created by a stool, skin, family or clan for indefinite duration; and (e) licenses and other lesser interests."
            },
            {
                "question": "How is customary land managed under the Land Act of Ghana?",
                "answer": "Under the Land Act, 2020 (Act 1036), customary land is managed by traditional authorities (stools, skins, families, and clans) in accordance with customary law. The Act requires the establishment of Customary Land Secretariats to manage stool, skin, and family lands. These secretariats maintain records of land transactions, facilitate land use planning, and promote transparency in land administration. Stool lands are administered by the Lands Commission on behalf of the stools."
            },
            {
                "question": "What is the role of the Lands Commission under Ghanaian land law?",
                "answer": "Under the Land Act, 2020 (Act 1036) and the Lands Commission Act, 2008 (Act 767), the Lands Commission is responsible for: (a) managing public lands and any lands vested in the President on behalf of the people of Ghana; (b) advising the Government, local authorities and traditional authorities on policy framework for the development of particular areas of Ghana; (c) formulating and submitting to Government recommendations on land policy; (d) processing applications for grants and land-related concessions; and (e) maintaining land registries."
            },
            {
                "question": "What are the provisions for land registration under the Land Act of Ghana?",
                "answer": "Under the Land Act, 2020 (Act 1036), all interests in land must be registered with the Lands Commission to be valid against third parties. The Act provides for a system of title registration where the register is conclusive evidence of ownership. Unregistered interests are enforceable only between the immediate parties. The Act also requires that all land transactions be in writing and witnessed to be valid."
            },
        ]
        qa_pairs.extend(land_law)
        
        # Companies Act, 2019 (Act 992)
        company_law = [
            {
                "question": "What is the Companies Act, 2019 (Act 992) of Ghana?",
                "answer": "The Companies Act, 2019 (Act 992) is Ghana's primary legislation governing the incorporation, registration, management, and winding up of companies. It replaced the Companies Act, 1963 (Act 179). The Act recognizes various types of companies including companies limited by shares, companies limited by guarantee, and unlimited companies. It establishes the Office of the Registrar of Companies and provides for corporate governance standards."
            },
            {
                "question": "What are the types of companies recognized under the Companies Act of Ghana?",
                "answer": "Under the Companies Act, 2019 (Act 992), the following types of companies are recognized: (a) company limited by shares - where the liability of members is limited to the amount unpaid on their shares; (b) company limited by guarantee - where liability is limited to the amount members undertake to contribute; (c) unlimited company - where members have unlimited liability; (d) external company - a company incorporated outside Ghana but operating in Ghana; and (e) one-person company - a company with only one member."
            },
            {
                "question": "What are the requirements for incorporating a company in Ghana?",
                "answer": "Under the Companies Act, 2019 (Act 992), to incorporate a company in Ghana, the incorporators must: (a) choose a unique company name and obtain clearance from the Registrar; (b) prepare the company's constitution; (c) file the incorporation documents with the Registrar including Form 3 (registration statement), the constitution, and particulars of directors and secretary; (d) pay the prescribed fees; and (e) obtain a certificate of incorporation from the Registrar. A company may also need to register for taxes and obtain relevant business licenses."
            },
            {
                "question": "What are the duties of directors under the Companies Act of Ghana?",
                "answer": "Under the Companies Act, 2019 (Act 992), directors owe fiduciary duties to the company including: (a) duty to act in good faith in the best interests of the company; (b) duty to exercise powers for proper purposes; (c) duty not to fetter discretion; (d) duty to avoid conflicts of interest; (e) duty not to accept benefits from third parties; (f) duty to declare interest in proposed transactions; and (g) duty to exercise reasonable care, skill and diligence. Directors who breach these duties may be personally liable."
            },
            {
                "question": "How can a company be wound up under the Companies Act of Ghana?",
                "answer": "Under the Companies Act, 2019 (Act 992), a company may be wound up: (a) voluntarily by members' resolution when the company has achieved its purpose or the period fixed for its duration has expired; (b) voluntarily by creditors when the company is insolvent; or (c) by the court on grounds including inability to pay debts, public interest, or just and equitable grounds. The winding up process involves appointing a liquidator, realizing assets, paying creditors, and distributing any surplus to members."
            },
        ]
        qa_pairs.extend(company_law)
        
        # Courts Act, 1993 (Act 459) and Judicial System
        courts_law = [
            {
                "question": "What is the structure of the Ghanaian court system?",
                "answer": "The Ghanaian court system comprises the Superior Courts and the Lower Courts. The Superior Courts include: the Supreme Court (the highest court and final appellate court), the Court of Appeal (hears appeals from the High Court), and the High Court (has unlimited original jurisdiction). The Lower Courts include: Circuit Courts, District Courts, and specialized tribunals. There are also Community Tribunals for minor matters and specialized courts like the Commercial Court, Financial and Economic Crimes Court, and the Human Rights Court."
            },
            {
                "question": "What is the jurisdiction of the Supreme Court of Ghana?",
                "answer": "Under Article 130 of the 1992 Constitution and the Courts Act, 1993, the Supreme Court of Ghana has: (a) exclusive original jurisdiction in constitutional matters and questions of constitutional interpretation; (b) appellate jurisdiction from judgments of the Court of Appeal; (c) power to review its own decisions; (d) supervisory jurisdiction over all courts and adjudicating authorities; and (e) power to issue writs including habeas corpus, certiorari, mandamus, and prohibition. The Supreme Court consists of not less than nine and not more than fifteen justices."
            },
            {
                "question": "What is the jurisdiction of the High Court of Ghana?",
                "answer": "Under Article 140 of the 1992 Constitution, the High Court has: (a) original jurisdiction in all matters, both civil and criminal, except those for which the Constitution provides differently; (b) jurisdiction to enforce fundamental human rights and freedoms; (c) power to issue writs including habeas corpus, certiorari, mandamus, and prohibition; and (d) appellate and supervisory jurisdiction over lower courts and tribunals. The High Court also has specialized divisions including the Commercial Division, Land Division, and Human Rights Division."
            },
        ]
        qa_pairs.extend(courts_law)
        
        # Marriage and Family Law
        family_law = [
            {
                "question": "What are the types of marriages recognized in Ghana?",
                "answer": "Ghana recognizes three types of marriages: (a) Customary marriage - conducted according to the customary law of the parties and registered under the Customary Marriage and Divorce (Registration) Law, 1985; (b) Ordinance marriage (also called statutory marriage) - a monogamous marriage under the Marriage Ordinance, 1884-1985, conducted in a church, mosque, or before a registrar; and (c) Islamic marriage - for Muslims, under Islamic law and customs. Each type has different requirements and legal effects, particularly regarding inheritance and property rights."
            },
            {
                "question": "What is the Intestate Succession Law of Ghana?",
                "answer": "The Intestate Succession Law, 1985 (PNDC Law 111) governs the distribution of property when a person dies without a valid will. Under this law, the deceased's estate is distributed as follows: the spouse receives the household chattels and the house where they resided; if there are children, the remainder is divided with 3/9 to the spouse, 3/9 to the children, 1/9 to the surviving parent, and 2/9 to the customary family. If there is no spouse, children inherit 3/4 and the customary family inherits 1/4. The law aims to protect the surviving spouse and children."
            },
            {
                "question": "What is the Matrimonial Causes Act of Ghana?",
                "answer": "The Matrimonial Causes Act, 1971 (Act 367) governs divorce proceedings in Ghana for statutory (ordinance) marriages. The sole ground for divorce is that the marriage has broken down beyond reconciliation, which may be proved by: adultery, unreasonable behavior, desertion for at least two years, living apart for at least two years (with consent), or living apart for at least five years. The court may also make orders regarding maintenance, custody of children, and division of property."
            },
        ]
        qa_pairs.extend(family_law)
        
        # Data Protection
        data_protection = [
            {
                "question": "What is the Data Protection Act of Ghana?",
                "answer": "The Data Protection Act, 2012 (Act 843) is Ghana's primary legislation on data protection and privacy. It establishes the Data Protection Commission and provides for the protection of the privacy of the individual and personal data. The Act regulates the collection, processing, storage, and use of personal data by data controllers. It requires data controllers to register with the Commission and to obtain consent for processing personal data. The Act also provides for the rights of data subjects including the right to access, correct, and delete personal data."
            },
            {
                "question": "What are the principles of data protection under the Data Protection Act of Ghana?",
                "answer": "Under the Data Protection Act, 2012 (Act 843), the principles of data protection include: (a) accountability - data controllers are responsible for compliance; (b) lawfulness - processing must have a lawful basis; (c) specification of purpose - data must be collected for a specified purpose; (d) data quality - data must be accurate and kept up to date; (e) openness - data subjects must be informed; (f) security safeguards - appropriate measures must be taken to protect data; and (g) data subject participation - individuals have rights over their data."
            },
        ]
        qa_pairs.extend(data_protection)
        
        # Environmental Law
        environmental_law = [
            {
                "question": "What is the Environmental Protection Agency Act of Ghana?",
                "answer": "The Environmental Protection Agency Act, 1994 (Act 490) establishes the Environmental Protection Agency (EPA) as the leading public body for protecting and improving the environment in Ghana. The EPA is responsible for: advising the Minister on environmental policy; coordinating activities of bodies with environmental functions; issuing environmental permits and pollution abatement notices; prescribing standards for air, water, and soil quality; and ensuring compliance with environmental regulations. The EPA also conducts environmental impact assessments."
            },
            {
                "question": "What are the requirements for Environmental Impact Assessment in Ghana?",
                "answer": "Under the Environmental Assessment Regulations, 1999 (LI 1652), any undertaking that is likely to have adverse effects on the environment must undergo an environmental impact assessment before commencement. The process involves: preliminary environmental report, scoping, detailed EIA study, public review, EPA review and decision, and environmental management plan. Activities requiring EIA include mining, manufacturing, infrastructure projects, and developments in environmentally sensitive areas. Failure to obtain environmental permits attracts penalties."
            },
        ]
        qa_pairs.extend(environmental_law)
        
        # Right to Information
        rti_law = [
            {
                "question": "What is the Right to Information Act of Ghana?",
                "answer": "The Right to Information Act, 2019 (Act 989) gives effect to Article 21(1)(f) of the 1992 Constitution which guarantees the right of access to information. The Act allows any person to request information from public institutions and some private bodies that receive public funding. Information must be provided within 14 days of request. Certain information is exempt from disclosure including national security information, personal privacy, and trade secrets. The Act establishes the Right to Information Commission to oversee implementation."
            },
            {
                "question": "How can one request information under the Right to Information Act of Ghana?",
                "answer": "Under the Right to Information Act, 2019 (Act 989), to request information: (a) submit a written or electronic application to the information officer of the institution; (b) describe the information sought with sufficient detail; (c) specify the form in which you want the information; and (d) pay any prescribed fees. The institution must respond within 14 days, extending to 21 days if needed. If the request is denied, the applicant may apply for internal review and then appeal to the Right to Information Commission or the courts."
            },
        ]
        qa_pairs.extend(rti_law)
        
        return qa_pairs
    
    def save_dataset(self, qa_pairs: List[Dict], filename: str):
        """Save the dataset to a JSONL file."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
                
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
        return output_path
    
    def run(self):
        """Run the full scraping and dataset generation pipeline."""
        print("=" * 60)
        print("Ghanaian Law Dataset Generator")
        print("=" * 60)
        
        # Generate constitution dataset
        print("\n1. Generating Ghana Constitution Q&A pairs...")
        constitution_qa = self.generate_constitution_dataset()
        print(f"   Generated {len(constitution_qa)} Q&A pairs from the Constitution")
        
        # Generate other laws dataset
        print("\n2. Generating Ghanaian Laws Q&A pairs...")
        laws_qa = self.generate_ghanaian_laws_dataset()
        print(f"   Generated {len(laws_qa)} Q&A pairs from various Ghanaian laws")
        
        # Combine all Q&A pairs
        all_qa = constitution_qa + laws_qa
        print(f"\n3. Total Q&A pairs generated: {len(all_qa)}")
        
        # Save to file
        print("\n4. Saving dataset...")
        output_file = self.save_dataset(all_qa, "ghanaian_law_comprehensive.jsonl")
        
        # Try online scraping if libraries are available
        if SCRAPING_AVAILABLE:
            print("\n5. Attempting to scrape additional content from online sources...")
            scraped_articles = self.scrape_ghana_constitution_online()
            if scraped_articles:
                print(f"   Scraped {len(scraped_articles)} additional articles")
                # Generate Q&A from scraped content
                scraped_qa = []
                for article in scraped_articles:
                    scraped_qa.extend(self._generate_qa_from_article({
                        "chapter": 0,
                        "article": article.get("article_number", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", "")
                    }))
                if scraped_qa:
                    self.save_dataset(scraped_qa, "ghanaian_law_scraped.jsonl")
        
        print("\n" + "=" * 60)
        print("Dataset generation complete!")
        print(f"Output file: {output_file}")
        print("=" * 60)
        
        return all_qa


def main():
    """Main entry point."""
    # Get the script's directory and set output to dataset folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "dataset")
    
    scraper = GhanaianLawScraper(output_dir=output_dir)
    scraper.run()


if __name__ == "__main__":
    main()
