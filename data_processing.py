import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.stats import anderson_ksamp, entropy
import re

def data_type(row):
    '''
    Converts data type from text to label.
    '''
    return {np.nan: 0, None: 0, "": 0, "media": 1, "newline": 2, "title": 3, "term": 4, "exercise": 5, "problem": 6,"solution": 7, "note": 8, "list": 9, "glossary": 10}[row]

def is_comparative(txt):
    '''
    Checks for comparative terms in the comment.
    '''
    txt = str(txt).lower()
    if re.search(r"as (\w+) as", txt) is not None:
        return True
    if re.search(r"is (\w+) than", txt) is not None:
        return True
    if re.search(r"are (\w+) than", txt) is not None:
        return True
    if "similar" in txt:
        return True
    if "compar" in txt:
        return True
    if "likewise" in txt:
        return True
    if "just as" in txt:
        return True
    if "same" in txt:
        return True
    if "differ" in txt:
        return True
    if "have in common" in txt:
        return True
    if "contrary" in txt:
        return True
    if "on the other hand" in txt:
        return True
    if "contrast" in txt:
        return True
    if "whereas" in txt:
        return True
    if "contrast" in txt:
        return True
    if "same as" in txt:
        return True
    if "is like" in txt:
        return True
    if "are like" in txt:
        return True
    if "relate" in txt:
        return True
    if "analog" in txt:
        return True
    if "correlat" in txt:
        return True
    if "equivalent" in txt:
        return True
    return False


def is_question(txt):
    '''
    Looks for question words and question marks. 
    '''
    for w in "who,where,when,why,what,which,how,?".split(","):
        if w in str(txt).lower(): return True
    return False


def is_elaboration(txt):
    '''
    Searches for elaboration words.
    '''
    for w in "because,since,cause,due to,owing to,therefore,consequently,as a result,thus,cause".split(","):
        if w in str(txt).lower(): return True
    return False


def parent_label(inp, df):
    '''
    Returns parent tag if a parent is present or -1 if there is no parent. Note that this is the actual parent ID. 
    '''
    if inp == -1 or pd.isnull(inp):
        return -1
    else:
        return df.loc[df["comment_id"] == inp]["predicted_label"].values[0]
def replies_count(inp, df):
    '''
    Finds number of replies to a comment using parent ID.
    '''
    return df.parent_id.tolist().count(inp)
def position_in_paragraph(row):
    '''
    Returns position in paragraph proportional to length of paragraph.
    '''
    marked_par = str(row["marked_par"])
    marked_text = str(row["marked_text"])
    return (marked_par.find(marked_text))/len(marked_par)
def position_in_document(para):
    '''
    Returns position in document proportional to length of document. 
    '''
    return max([((chapter_14.find(para))/len(chapter_14)),((chapter_15.find(para))/len(chapter_15)),((chapter_16.find(para))/len(chapter_16))])
def calculate_paragraph_ce(df, para):
    '''
    Return average CE for a given paragraph by matching marked paragraph entries. 
    '''
    p_comments = df.loc[df["marked_par"] == para]
    if p_comments.shape[0] ==0:
        return 0
    p_ce = sum(p_comments["predicted_label"])/p_comments.shape[0]
    return p_ce
        
def calculate_overlap_ce(df, location):
    '''
    Finds text overlaps using location ID and returns average CE. 
    '''
    overlap_comments = df.loc[df["location_id"] == location]
    if overlap_comments.shape[0] ==0:
        return 0
    overlap = sum(overlap_comments["predicted_label"])/overlap_comments.shape[0]
    return overlap
def get_color(inp):
    '''
    Calculates hue using formula from 2017 L@S paper. 
    '''
    inp += 3
    if inp>=1 and inp <4:
        inp =inp - 0.6
    hue = (inp*24)
    return hue
def heatmap_difference(df,location, is_true):
    '''
    Thread-level heatmap value.
    '''
    overlap_comments = df.loc[df["location_id"] == location]
    if overlap_comments.shape[0] ==0:
        return 0
    if is_true:
        return get_color(sum(overlap_comments["label"])/overlap_comments.shape[0])
    else:
        return get_color(sum(overlap_comments["predicted_label"])/overlap_comments.shape[0])
def generate_heatmap_par(dataframe):
    '''
    Paragraph level heatmap value.
    '''
    dataframe["paragraph_heat"] = 0
    paras = dataframe["marked_par"].unique()
    for item in range(len(paras)):
        i = paras[item]
        d_y= dataframe["label"][dataframe["marked_par"] == i]
        comments = dataframe["text"][dataframe["marked_par"] == i]
        # print(d_y)
        avg_truth = d_y.mean()
        t_color = get_color(avg_truth)
        dataframe["paragraph_heat"][dataframe["marked_par"] == i] = t_color

def generate_paragraph_entropy(dataframe):
    '''
    Find entropy for a given paragraph based on labels.
    '''
    dataframe["paragraph_entropy"] = 0
    paras = dataframe["marked_par"].unique()
    for item in range(len(paras)):
        i = paras[item]
        d_y= dataframe["label"][dataframe["marked_par"] == i]
        counts = d_y.value_counts()
        print(counts, entropy(counts))
        dataframe["paragraph_entropy"][dataframe["marked_par"] == i] = entropy(counts)

def get_hashtag(tag, body):
    '''
    Checks if a hashtag is present in the comment.
    '''
    print(body, type(body))
    if not isinstance(body, str) or body.count("#") == 0 or body.count(tag) ==0:
        return 0
    else:
        tokens = word_tokenize(body)
        ind = [i for i, x in enumerate(tokens) if x == tag]
        for i in ind:
            if tokens[i-1] == '#':
                return 1
        return 0

# chapter_5 = '''Section Overview
# Proteins are a class of macromolecules that perform a diverse range of functions for the cell. They help in metabolism by providing structural support and by acting as enzymes, carriers, or hormones. The building blocks of proteins (monomers) are amino acids. Each amino acid has a central carbon that is linked to an amino group, a carboxyl group, a hydrogen atom, and an R group or side chain. There are 20 commonly occurring amino acids, each of which differs in the R group. Each amino acid is linked to its neighbors by a peptide bond. A long chain of amino acids is known as a polypeptide.



# Proteins are organized at four levels: primary, secondary, tertiary, and (optional) quaternary. The primary structure is the unique sequence of amino acids. The local folding of the polypeptide to form structures such as the α helix and β-pleated sheet constitutes the secondary structure. The overall three-dimensional structure is the tertiary structure. When two or more polypeptides combine to form the complete protein structure, the configuration is known as the quaternary structure of a protein. Protein shape and function are intricately linked; any change in shape caused by changes in temperature or pH may lead to protein denaturation and a loss in function.
# Amino Acid Structure
# Amino acids are the monomers that make up proteins. Each amino acid has the same fundamental structure, which consists of a central carbon atom, also known as the alpha (α) carbon, bonded to an amino group (NH2), a carboxyl group (COOH), and to a hydrogen atom. Every amino acid also has another atom or group of atoms bonded to the central atom known as the R group (Figure). For an introduction on amino acids, click here for a short (4 minute) video.

# The molecular structure of an amino acid is given. An amino acid has an alpha carbon to which an amino group, a carboxyl group, a hydrogen, and a side chain are attached. The side chain varies for different amino acids, and is designated with an “R.”
# Amino acids have a central asymmetric carbon to which an amino group, a carboxyl group, a hydrogen atom, and a side chain (R group) are attached.

# Possible Discussion:
# Recall that one of the learning goals for this class is that you (a) be able to recognize in a molecular diagram the backbone of an amino acid and its side chain (R-group) and (b) that you be able to draw a generic amino acid. Make sure that you practice both. You should be able to recreate something like Figure 2 from memory.

# Using figure 2, which of the following is true about amino acids:

# amino acids contain polar functional groups
# amino acids contain basic functional groups
# amino acids contain acidic functional groups
# amino acids contain a variable group that can be either polar or nonpolar
# all of the above

# The Amino Acid Backbone
# The name "amino acid" is derived from the fact that all amino acids contain both an amino group and carboxyl-acid-group in their backbone. There are 20 amino acids present in proteins and each of these contain the same backbone. The backbone, when ignoring the hydrogen atoms, consists of the pattern:


# N-C-C

# When looking at a chain of amino acids it is always helpful to first orient yourself by finding the backbone pattern from the N terminus (the amino end of the first amino acid) to the C terminus (the carboxylic acid end of the last amino acid).
# The formation of a peptide bond between two amino acids is shown. When the peptide bond forms, the carbon from the carbonyl group becomes attached to the nitrogen from the amino group. The OH from the carboxyl group and an H from the amino group form a molecule of water..
# Peptide bond formation is a dehydration synthesis reaction. The carboxyl group of the first amino acid is linked to the amino group of the second incoming amino acid. In the process, a molecule of water is released and a peptide bond is formed.

# Try finding the backbone in the dipeptide formed from this reaction. The pattern you are looking for is: N-C-C-N-C-C

# Source: Bis2A original image
# The sequence and the number of amino acids ultimately determine the protein's shape, size, and function. Each amino acid is attached to another amino acid by a covalent bond, known as a peptide bond, which is formed by a dehydration reaction. The carboxyl group of one amino acid and the amino group of the incoming amino acid combine, releasing a molecule of water. The resulting bond is the peptide bond.

# Where do peptide bonds form?

# between the carboxyl group of one amino acid and the amino group of another amino acid
# between the adjacent carboxyl groups on amino acids
# between amino acids of two different protein chains
# between adjacent amino groups on amino acids
# a and c
# c and d

# Amino Acid R group
# The amino acid R group is a term that refers to the variable group on each amino acid. The amino acid backbone is identical on all amino acids, the R groups is different on all amino acids. For the structure of each amino acid refer to the figure below.

# The molecular structures of the twenty amino acids commonly found in proteins are given. These are divided into five categories: nonpolar aliphatic, polar uncharged, positively charged, negatively charged, and aromatic. Nonpolar aliphatic amino acids include glycine, alanine, valine, leucine, methionine, and isoleucine. Polar uncharged amino acids include serine, threonine, cysteine, proline, asparagine, and glutamine. Positively charged amino acids include lysine, arginine, and histidine. Negatively charged amino acids include aspartate and glutamate. Aromatic amino acids include phenylalanine, tyrosine, and tryptophan.
# There are 20 common amino acids commonly found in proteins, each with a different R group (variant group) that determines its chemical nature.
# POSSIBLE DISCUSSION:
# Let's think about the relevance of having 20 different amino acids. If you were using biology to build proteins from scratch, how might it be useful if you had 10 more different amino acids at your disposal? By the way, this is actually happening in a variety of research labs - why would this be potentially useful?
# Each variable group on an amino acid gives that amino acid specific chemical properties (acidic, basic, polar, or nonpolar). Many of the functional groups in the R groups have been visited in the module 2.1. Briefly, if there is a polar functional group on the R group of the amino acid, that amino acid is considered a polar amino acid. Likewise, if the R group contains only nonpolar covalent bonds then that amino acid is considered nonpolar.


# For example, amino acids such as valine, methionine, and alanine are nonpolar or hydrophobic in nature, while amino acids such as serine and threonine are polar and have hydrophilic side chains.

# Try identifying each amino acid in the figure above as either polar or nonpolar.
# Different Representations of Amino Acids
# In this class expect to see amino acids represented in different ways. Sometimes they will be drawn out to show their lewis structure, they can also be represented as "beads on a string" or by a single upper case letter or a three-letter abbreviation. For example, valine is known by the letter V or the three-letter symbol val.

# Protein Folding and Structure
# The shape of a protein is critical to its function. For example, an enzyme can bind to a specific substrate at a site known as the active site. If this active site is altered because of local changes or changes in overall protein structure, the enzyme may be unable to bind to the substrate. To understand how the protein gets its final shape or conformation, we need to understand the four levels of protein structure: primary, secondary, tertiary, and quaternary. For a short (4 minutes) introduction video on protein structure click here.

# Primary Structure
# The unique sequence of amino acids in a polypeptide chain is its primary structure. The linear sequence of amino acids in the polypeptide chain are held together by peptide bonds and result in the N-C-C-N-C-C patterned backbone. The primary structure is coded for in the DNA, a process you will learn about in the Transcription and Translation modules.

# The beads on a string structure of the primary protein sequence is depicted
# The primary structure of a protein is depicted here as "beads on a string" with the N terminus and C terminus labeled. The order in which you would read this peptide chain would begin with the N-terminus as Glycine, Isoleucine, etc and end with methionine.

# Source: Erin Easlon (own work)
# Secondary structure
# The local folding of the polypeptide in some regions gives rise to the secondary structure of the protein. The most common shapes created by secondary folding are the α-helix and β-pleated sheet structures. These secondary structures are held together by hydrogen bonds forming between the backbones of amino acids in close proximity to one another. More specifically, the oxygen atom in the carboxyl group from one amino acid can form a hydrogen bond with a hydrogen atom bound to the nitrogen in the amino group of another amino acid that is four amino acids farther along the chain.

# The illustration shows an alpha helix protein structure, which coils like a spring, and a beta-pleated sheet structure, which forms flat sheets stacked together. In an alpha-helix, hydrogen bonding occurs between the carbonyl group of one amino acid and the amino group of the amino acid that occurs four residues later. In a beta-pleated sheet, hydrogen bonding occurs between two different lengths of peptide that are antiparallel to one another.
# The α-helix and β-pleated sheet are secondary structures of proteins that form because of hydrogen bonding between carbonyl and amino groups in the peptide backbone. Certain amino acids have a propensity to form an α-helix, while others have a propensity to form a β-pleated sheet.

# Tertiary Structure
# The unique three-dimensional structure of a polypeptide is its tertiary structure. This structure is in part due to chemical interactions at work on the polypeptide chain. Primarily, the interactions among R groups creates the complex three-dimensional tertiary structure of a protein. The nature of the R groups found in the amino acids involved can counteract the formation of the hydrogen bonds described for standard secondary structures. For example, R groups with like charges are repelled by each other and those with unlike charges are attracted to each other (ionic bonds). When protein folding takes place, the hydrophobic R groups of nonpolar amino acids lay in the interior of the protein, whereas the hydrophilic R groups lay on the outside. These types of interactions are also known as hydrophobic interactions. Interaction between cysteine side chains forms disulfide linkages in the presence of oxygen, the only covalent bond forming during protein folding.

# This illustration shows a polypeptide backbone folded into a three-dimensional structure. Chemical interactions between amino acid side chains maintain its shape. These include an ionic bond between an amino group and a carboxyl group, hydrophobic interactions between two hydrophobic side chains, a hydrogen bond between a hydroxyl group and a carbonyl group, and a disulfide linkage.
# The tertiary structure of proteins is determined by a variety of chemical interactions. These include hydrophobic interactions, ionic bonding, hydrogen bonding and disulfide linkages. This image shows a flattened representation of a protein folded in tertiary structure. Without flattening, this protein would be a globular 3D shape.
# All of these interactions, weak and strong, determine the final three-dimensional shape of the protein. When a protein loses its three-dimensional shape, it may no longer be functional.

# Quaternary Structure
# In nature, some proteins are formed from several polypeptides, also known as subunits, and the interaction of these subunits forms the quaternary structure. Weak interactions between the subunits help to stabilize the overall structure. For example, a multisubunit protein called insulin (a globular protein) has a combination of hydrogen bonds and disulfide bonds that hold the multiple subunits together. Each of these subunits went through primary, secondary and tertiary folding independently of one another.

# Shown are the four levels of protein structure. The primary structure is the amino acid sequence. Secondary structure is a regular folding pattern due to hydrogen bonding. Two types of secondary structure are shown: a beta pleated sheet, which is flat with regular ripples, and an alpha helix, which coils like a spring. Tertiary structure is the three-dimensional folding pattern of the protein due to interactions between amino acid side chains. Quaternary structure is the interaction of two or more polypeptide chains.
# The four levels of protein structure can be observed in these illustrations.

# Source: modification of work by National Human Genome Research Institute
# Denaturation and Protein Folding
# Each protein has its own unique sequence and shape that are held together by chemical interactions. If the protein is subject to changes in temperature, pH, or exposure to chemicals, the protein structure may change, losing its shape without losing its primary sequence in what is known as denaturation. Denaturation is often reversible because the primary structure of the polypeptide is conserved in the process if the denaturing agent is removed, allowing the protein to resume its function. Sometimes denaturation is irreversible, leading to loss of function. One example of irreversible protein denaturation is when an egg is fried. The albumin protein in the liquid egg white is denatured when placed in a hot pan. Not all proteins are denatured at high temperatures; for instance, bacteria that survive in hot springs have proteins that function at temperatures close to boiling. The stomach is also very acidic, has a low pH, and denatures proteins as part of the digestion process; however, the digestive enzymes of the stomach retain their activity under these conditions.

# Protein folding is critical to its function. It was originally thought that the proteins themselves were responsible for the folding process. Only recently was it found that often they receive assistance in the folding process from protein helpers known as chaperones (or chaperonins) that associate with the target protein during the folding process. They act by preventing aggregation of polypeptides that make up the complete protein structure, and they disassociate from the protein once the target protein is folded.

# FOR ADDITIONAL INFORMATION:
# For an additional perspective on proteins, view this animation called “Biomolecules: The Proteins.”

# Khan Academy link
# Protein structure.

# Which of the following changes when a protein denatures?

# amino acid sequence
# length of the entire protein
# three dimensional structure
# the peptide bonds between the amino acids
# a and d
# b and d

# Which categories of amino acid would you expect to find on the surface of a soluble protein, and which would you expect to find in the interior? What distribution of amino acids would you expect to find in a protein embedded in a lipid bilayer?


# Describe the differences in the four stages of protein folding.


# Bis2A: 2.2 pH and pKA
# Section Overview
# The pH of a solution is a value related to the hydrogen ion concentration and is one of many chemical characteristics that is highly regulated in living organisms. Acids and bases can change pH values and buffers are compounds that tend to moderate changes in pH in solution.

# What is the role of Acid/Base Chemistry in Bis2A?
# We have learned that the behavior of chemical functional groups depend greatly on the composition, order and properties of their constituent atoms. As we will see, some of the properties of key biological functional groups can be altered depending on the pH (hydrogen ion concentration) of the solution that they are bathed in. For example, some of the functional groups on the amino acid molecules that make up proteins can exist in different chemical states depending on the pH. We will learn that the chemical state of these functional groups in the context of a protein can have have a profound effect on the shape of protein or its ability to carry out chemical reactions. As we move through the course we will see numerous examples of this type of chemistry in different contexts.

# pH
# pH is formally defined as:


#  Equation defining pH

# In the equation above, the square brackets surrounding H+ indicate concentration.


# If necessary try a math review at wiki logarithm or kahn logarithm.


# Also see: concentration dictionary or wiki concentration.
# Hydrogen ions are spontaneously generated in pure water by the dissociation (ionization) of a small percentage of water molecules into equal numbers of hydrogen (H+) ions and hydroxide (OH-) ions. While the hydroxide ions are kept in solution by their hydrogen bonding with other water molecules, the hydrogen ions, consisting of naked protons, are immediately attracted to un-ionized water molecules, forming hydronium ions (H30+). Still, by convention, scientists refer to hydrogen ions and their concentration as if they were free in this state in liquid water. This is another example of a shortcut that we often take - it's easier to write H+ rather than H3O+. We just need to realize that this shortcut is being taken, else confusion will ensue.

# Water molecule dissosiated into a hydroxyl group and a proton. The proton will then recombine with a water molecule.
# Water spontaneously dissociates into a proton and hydroxyl group. The proton will combine with a water molecule forming a hydronium ion.

# Source: http://www.biologycorner.com/worksheets/acids_bases_coloring.html
# pH of a solution is a measure of the concentration of hydrogen ions in a solution (or the number of hydronium ions). The number of hydrogen ions is a direct measure of how acidic or how basic a solution is. The pH scale is logarithmic and ranges from 0 to 14 (Figure). We define pH=7.0 as neutral. Anything with a pH below 7.0 is termed acidic and any reported pH above 7.0 is termed alkaline or basic. Extremes in pH in either direction from 7.0 are usually considered inhospitable to life, though examples exist to the contrary. pH in the human body usually ranges between 6.8 and 7.4, except in the stomach where the pH is typically between 1 and 2.

# Watch this video for a straightforward explanation of pH and its logarithmic scale.

# The scale of pH ranging from acidic to basic with different biological examples of compounds or substances that exist at that particular pH.
# The pH scale ranging from acidic to basic with various biological compounds or substances that exist at that particular pH.

# Source: https://en.wikipedia.org/wiki/PH
# For additional information:
# Watch this video for an alternative explanation of pH and its logarithmic scale.

# The concentration of hydrogen ions dissociating from pure water is 1 × 10-7 moles H+ ions per liter of water. One Mole (mol) of a substance (which can be atoms, molecules, ions, etc), is defined as being equal to 6.02 x 1023 particles of the substance. Therefore, 1 mole of water is equal to 6.02 x 1023 water molecules. The pH is calculated as the negative of the base 10 logarithm of this unit of concentration. The log10 of 1 × 10-7 is -7.0, and the negative of this number yields a pH of 7.0, which is also known as neutral pH.

# Non-neutral pH readings result from dissolving acids or bases in water. High concentrations of hydrogen ions yields a low pH number, whereas low levels of hydrogen ions result in a high pH. This inverse relationship between pH and the concentration of protons confuses many students - take the time to convince yourself that you "get it". An acid is a substance that increases the concentration of hydrogen ions (H+) in a solution, usually by having one of its hydrogen atoms dissociate. For example, we have learned that the carboxyl functional group is an acid. The hydrogen atom can dissociate from the oxygen atom resulting in a free proton and a negatively charged functional group. A base provides either hydroxide ions (OH–) or other negatively charged ions that combine with hydrogen ions, effectively reducing the H+ concentration in the solution and thereby raising the pH. In cases where the base releases hydroxide ions, these ions bind to free hydrogen ions, generating new water molecules. For example, we have learned that the amine functional group is a base. The nitrogen atom will accept hydrogen ions in solution, thereby reducing the number of hydrogen ions which raises the pH of the solution.

# The carboxyl group in a protonated and deprotonated state as an acid example. The amino group in a protonated and deprotonated state as a base example.
# The carboxylic acid group acts as an acid by releasing a proton into solution. This increases the number of protons in solution and thus decreases the pH. The amino group acts as a base by accepting hydrogen ions from solution, decreasing the number of hydrogen ions in solutions, thus increasing the pH.

# Source: Created by Erin Easlon
# So how can organisms whose bodies require a near-neutral pH ingest acidic and basic substances (a human drinking orange juice, for example) and survive? Buffers are the key. Buffers readily absorb excess H+ or OH–, keeping the pH of the body carefully maintained in the narrow range required for survival. Maintaining a constant blood pH is critical to a person’s well-being. The buffer maintaining the pH of human blood involves carbonic acid (H2CO3), bicarbonate ion (HCO3–), and carbon dioxide (CO2). When bicarbonate ions combine with free hydrogen ions and become carbonic acid, hydrogen ions are removed, moderating pH changes. Similarly, as shown in [link], excess carbonic acid can be converted to carbon dioxide gas and exhaled through the lungs. This prevents too many free hydrogen ions from building up in the blood and dangerously reducing the blood’s pH. Likewise, if too much OH– is introduced into the system, carbonic acid will combine with it to create bicarbonate, lowering the pH. Without this buffer system, the body’s pH would fluctuate enough to put survival in jeopardy.

# An H2O molecule can combine with a CO2 molecule to form H2CO3, or carbonic acid. A proton may dissociate from H2CO3, forming bicarbonate, or HCO3-, in the process. The reaction is reversible so that if acid is added protons combined with bicarbonate to form carbonic acid.
# This diagram shows the body’s buffering of blood pH levels. The blue arrows show the process of raising pH as more CO2 is made. The purple
# Other examples of buffers are antacids used to combat excess stomach acid. Many of these over-the-counter medications work in the same way as blood buffers, usually with at least one ion capable of absorbing hydrogen and moderating pH, bringing relief to those that suffer “heartburn” after eating. The unique properties of water that contribute to this capacity to balance pH—as well as water’s other characteristics—are essential to sustaining life on Earth.

# Here are some additional links on pH and pKa to help learn the material. Note that there is an additional module devoted to pKa.

# Chemwiki Links
# Determining and calculating pH
# Acid-Base titrations
# pH and pKa
# Khan Academy Links
# What is pH
# strong acids and bases
# weak acids
# weak acid-base equilibria
# pH and pKa
# Simulations
# Acid-base simulation.
# Intro to acids, bases, pH.
# pKa
# pKa is defined as the negative log10 of the dissociation constant of an acid, its Ka. Therefore, the pKa is a quantitative measure of how easily or how readily the acid gives up its proton [H+] in solution and thus a measure of the "strength" of the acid . Strong acids have a small pKa, weak acids have a larger pKa.

# The most common acid we will talk about in Bis2A is the carboxylic acid functional group. These acids are typically weak acids, meaning that they only partially dissociate (into H+ cations and RCOO- anions) in neutral solution. HCL (hydrogen chloride) is a common strong acid, meaning that it will fully dissociate into H+ and Cl-.

# An example of a strong base HCL dissociating in water. An example of a strong base sodium hydroxide dissociating in water. An example of a weak acid (acetic acid) and a weak base (amminium) dissociating in water. The pKa value is shown on the left.
# An example of strong acids and strong bases in their protonation and deprotonation states. The value of their pKa is shown on the left.

# Source: https://www.boundless.com/chemistry/textbooks/boundless-chemistry-textbook/acids-and-bases-15/strength-of-acids-109/strong-acids-455-6873/
# Electronegativity plays a role in acid strength. The greater electronegativity of the atom or atoms attached to the H-O group in the acid results in a weaker H-O bond, which is thus more readily ionized. This means that the pull on the electrons away from the hydrogen atom gets greater when the oxygen atom attached to the hydrogen atom is also attached to another electronegative atom. An example of this is HOCL. The electronegative CL polarizes the H-O bond, weakening it and facilitating the ionization of the hydrogen. If we compare this to a weak acid where the oxygen is bound to a carbon atom (as in carboxylic acids) the oxygen is bound to the hydrogen and carbon atom. In this case, the oxygen is not bound to another electronegative atom. Thus the H-O bond is not further destabilized and the acid is considered a weak acid (it does not give up the proton as easily as a strong acid).

# Acetic acid with dipole moments on the C, O and H. Hypoochlorous acid with dipole moments on the O, H and CL. Arrows are drawn from the positive dipole moments towards the negative dipole moments. Oxygen and Cloride have the negative dipole moments, and carbon and hydrogen have positive dipole moments.
# The strength of the acid can be determined by the electronegativity of the atom the oxygen is bound to. For example, the weak acid Acetic Acid, the oxygen is bound to carbon, an atom with low electronegativity. In the strong acid, Hypochlorous acid, the oxygen atom is bound to an even more electronegative Chloride atom.

# Source: Created by Erin Easlon
# In Bis2A you are going to be asked to relate pH and pKa to each other when discussing the protonation state of an acid or base, for example, in amino acids. How can we use the information given in this module to answer the question: Will the functional groups on the amino acid Glutamic Acid be protonated or deprotonated at a pH of 2, at a pH of 8, at a pH of 11?

# In order to start answering this question we need to create a relationship between pH and pKa. The relationship between pKa and pH is mathematically represented by Henderson-Hasselbach equation shown below, where [A-] represents the deprotonated form of the acid and [HA] represents the protonated form of the acid.

# The henderson-hasselbach equation where ph equals the pka plus the log of the deprotonated state divided by the protonated state of the acid.
# The Henderson-Hasselbach equation.
# A solution to this equation is obtained by setting pH = pKa. In this case, log([A-] / [HA]) = 0, and [A-] / [HA] = 1. This means that when the pH is equal to the pKa there are equal amounts of protonated and deprotonated forms of the acid. For example, if the pKa of the acid is 4.75, at a pH of 4.75 that acid will exist as 50% protonated and 50% deprotonated. This also means that as the pH rises, more of the acid will be converted into the deprotonated state and at some point the pH will be so high that the majority of the acid will exist in the deprotonated state.

# A graph of the protonation and deprotonation state of acetic acid at different pHs. The X axis is OH equivalent, the Y axis is pH. The pKa is indicated at 4.75. At 4.75 the protonation and deprotonation state of acetic acid is equal or considered 50/50. Below the pKa the acid is protonated, above the pKa the acid is deprotonated. 
# This graph depicts the protonation state of acetic acid as the pH changes. At a pH below the pKa, the acid is protonated. At a pH above the pKa the acid is deprotonated. If the pH equals the pKa, the acid is 50% protonated and 50% deprotonated.

# Source: Created by Erin Easlon
# In Bis2A we will be looking at the protonation state and deprotonation state of amino acids. Amino acids contain multiple functional groups that can be acids or bases. Therefore their protonation/deprotonation status can be more complicated. Below is the relationship between the pH and pKa of the amino acid Glutamic Acid. In this graph we can ask the question we posed earlier: Will the functional groups on the amino acid Glutamic Acid be protonated or deprotonated at a pH of 2, at a pH of 8, at a pH of 11?

# A graph of the protonation and deprotonation state of glutamic acid at different pHs. The X axis is OH equivalent, the Y axis is pH. The multiple pKas are indicated at pH 2.2, 4.04, and 9.7. 
# This graph depicts the protonation state of glutamic acid as the pH changes. At a pH below the pKa for each functional group on the amino acid, the functional group is protonated. At a pH above the pKa for the functional group it is deprotonated. If the pH equals the pKa, the functional group is 50% protonated and 50% deprotonated.

# Source: Created by Erin Easlon
# Prep for the Test:
# What is the overall charge of Glutamic acid at a pH of 5?
# What is the overall charge of Glutamic acid at a pH of 10?
# What is the charge on a base when it is protonated?
# What is the charge on a base when it is deprotonated?
# What is the charge on an acid when it is protonated?
# What is the charge on an acid when it is deprotonated?
# Review Questions
# Which of the following statements is not true?

# Water is polar.
# Water stabilizes temperature.
# Water is essential for life.
# Water is the most abundant molecule in the Earth’s atmosphere.

# When acids are added to a solution, the pH should ________.

# decrease
# increase
# stay the same
# cannot tell without testing

# A molecule that binds up excess hydrogen ions in a solution is called a(n) ________.

# acid
# isotope
# base
# donator

# Which of the following statements is true?

# Acids and bases cannot mix together.
# Acids and bases will neutralize each other.
# Acids, but not bases, can change the pH of a solution.
# Acids donate hydroxide ions (OH–); bases donate hydrogen ions (H+).

# Chemwiki Links
# Determining and calculating pH
# Acid-Base titrations
# pH and pKa
# Khan academy links:
# What is pH
# strong acids and bases
# Simulations
# Acid-base simulation.
# Intro to acids, bases, pH.
# i    
# ''''''###chapter 5


chapter_14 = '''Skip to main content
Help Spread the Word! The LibreTexts Project is the now the highest ranked and most visited online OER textbook project thanks to you. 
libretexts_section_complete_photon_124.png
Chemistry Biology Geosciences Mathematics Statistics Physics Social Sciences Engineering Medicine Business Photosciences Humanities
Search  
How can we help you?
 Search
 
Username
   
Password
   Sign in
 
Sign in
  Expand/collapse global hierarchy  Home   Course LibreTexts   University of California Davis   BIS 2A: Introductory Biology (Facciotti)   Readings   S2018_Lecture_Readings   Expand/collapse global location
S2018_Lecture14_Reading
Last updatedJul 24, 2018
S2018_Lecture13_Reading
 
S2018_Lecture15_Reading
picture_as_pdf
Donate
Oxidation of Pyruvate and the TCA Cycle
Overview of Pyruvate Metabolism and the TCA Cycle

Under appropriate conditions, pyruvate can be further oxidized. One of the most studied oxidation reactions involving pyruvate is a two-part reaction involving NAD+ and molecule called co-enzyme A, often abbreviated simply as "CoA". This reaction oxidizes pyruvate, leads to a loss of one carbon via decarboxylation, and creates a new molecule called acetyl-CoA. The resulting acetyl-CoA can enter several pathways for the biosynthesis of larger molecules or it can be routed to another pathway of central metabolism called the Citric Acid Cycle, sometimes also called the Krebs Cycle, or Tricarboxylic Acid (TCA) Cycle. Here the remaining two carbons in the acetyl group can either be further oxidized or serve again as precursors for the construction of various other molecules. We discuss these scenarios below. 

The different fates of pyruvate and other end products of glycolysis
The glycolysis module left off with the end-products of glycolysis: 2 pyruvate molecules, 2 ATPs and 2 NADH molecules. This module and the module on fermentation explore what the cell can do with the pyruvate, ATP and NADH that were generated. 

The fates of ATP and NADH
In general, ATP can be used for or coupled to a variety of cellular functions including biosynthesis, transport, replication etc. We will see many such examples throughout the course.  

What to do with the NADH however, depends on the conditions under which the cell is growing. In some cases, the cell will opt to rapidly recycle NADH back into to NAD+. This occurs through a process called fermentation in which the electrons initially taken from the glucose derivatives are returned to more downstream products via another red/ox transfer (described in more detail in the module on fermentation). Alternatively, NADH can be recycled back into NAD+ by donating electrons to something known as an electron transport chain (this is covered in the module on respiration and electron transport). 

The fate of cellular pyruvate
Pyruvate can be used as a terminal electron acceptor (either directly or indirectly) in fermentation reactions, and is discussed in the fermentation module. 
Pyruvate could be secreted from the cell as a waste product. 
Pyruvate could be further oxidized to extract more free energy from this fuel.
Pyruvate can serve as a valuable intermediate compound linking some of the core carbon processing metabolic pathways
The further oxidation of pyruvate
In respiring bacteria and archaea, the pyruvate is further oxidized in the cytoplasm. In aerobically respiring eukaryotic cells, the pyruvate molecules produced at the end of glycolysis are transported into mitochondria, which are sites of cellular respiration and house oxygen consuming electron transport chains (ETC in module on respiration and electron transport). Organisms from all three domains of life share similar mechanisms to further oxidize the pyruvate to CO2. First pyruvate is decarboxylated and covalently linked to co-enzyme A via a thioester linkage to form the molecule known as acetyl-CoA. While acetyl-CoA can feed into multiple other biochemical pathways we now consider its role in feeding the circular pathway known as the Tricarboxylic Acid Cycle, also referred to as the TCA cycle, the Citric Acid Cycle or the Krebs Cycle. This process is detailed below.

Conversion of Pyruvate into Acetyl-CoA
In a multi-step reaction catalyzed by the enzyme pyruvate dehydrogenase, pyruvate is oxidized by NAD+, decarboxylated, and covalently linked to a molecule of co-enzyme A via a thioester bond. The release of the carbon dioxide is important here, this reaction often results in a loss of mass from the cell as the CO2 will diffuse or be transported out of the cell and become a waste product. In addition, one molecule of NAD+ is reduced to NADH during this process per molecule of pyruvate oxidized.  Remember: there are two pyruvate molecules produced at the end of glycolysis for every molecule of glucose metabolized; thus, if both of these pyruvate molecules are oxidized to acetyo-CoA two of the original six carbons will be converted to waste.

SUGGESTED DISCUSSION

We have already discussed the formation of a thioester bond in another unit and lecture. Where was this specifically? What was the energetic significance of this bond? What are the similarities and differences between this example (formation of thioester with CoA) and the previous example of this chemistry?

 



Figure 1. Upon entering the mitochondrial matrix, a multi-enzyme complex converts pyruvate into acetyl CoA. In the process, carbon dioxide is released and one molecule of NADH is formed.
SUGGESTED DISCUSSION

Describe the flow and transfer of energy in this reaction using good vocabulary - (e.g. reduced, oxidized, red/ox, endergonic, exergonic, thioester, etc. etc.). You can peer edit - someone can start a description, another person can make it better, another person can improve it more etc. . .

 

In the presence of a suitable terminal electron acceptor, acetyl CoA delivers (exchanges a bond) its acetyl group to a four-carbon molecule, oxaloacetate, to form citrate (designated the first compound in the cycle). This cycle is called by different names: the citric acid cycle (for the first intermediate formed—citric acid, or citrate), the TCA cycle (since citric acid or citrate and isocitrate are tricarboxylic acids), and the Krebs cycle, after Hans Krebs, who first identified the steps in the pathway in the 1930s in pigeon flight muscles.

The Tricarboxcylic Acid (TCA) Cycle
In bacteria and archaea reactions in the TCA cycle typically happen in the cytosol. In eukaryotes, the TCA cycle takes place in the matrix of mitochondria. Almost all (but not all) of the enzymes of the TCA cycle are water soluble (not in the membrane), with the single exception of the enzyme succinate dehydrogenase, which is embedded in the inner membrane of the mitochondrion (in eukaryotes). Unlike glycolysis, the TCA cycle is a closed loop: the last part of the pathway regenerates the compound used in the first step. The eight steps of the cycle are a series of red/ox, dehydration, hydration, and decarboxylation reactions that produce two carbon dioxide molecules, one ATP, and reduced forms of NADH and FADH2.  



Figure 2. In the TCA cycle, the acetyl group from acetyl CoA is attached to a four-carbon oxaloacetate molecule to form a six-carbon citrate molecule. Through a series of steps, citrate is oxidized, releasing two carbon dioxide molecules for each acetyl group fed into the cycle. In the process, three NAD+ molecules are reduced to NADH, one FAD+ molecule is reduced to FADH2, and one ATP or GTP (depending on the cell type) is produced (by substrate-level phosphorylation). Because the final product of the TCA cycle is also the first reactant, the cycle runs continuously in the presence of sufficient reactants.

Attribution: “Yikrazuul”/Wikimedia Commons (modified)

 

NOTE

We are explicitly making reference to eukaryotes, bacteria and archaea when we discuss the location of the TCA cycle because many beginning students of biology tend to exclusively associate the TCA cycle with mitochondria.  Yes, the TCA cycle occurs in the mitochondria of eukaryotic cells.  However, this pathway is not exclusive to eukaryotes; it occurs in bacteria and archaea too!  

Steps in the TCA Cycle
Step 1: 
The first step of the cycle is a condensation reaction involving the two-carbon acetyl group of acetyl-CoA with one four-carbon molecule of oxaloacetate. The products of this reaction are the six-carbon molecule citrate and free co-enzyme A. This step is considered irreversible because it is so highly exergonic. Moreover, the rate of this reaction is controlled through negative feedback by ATP. If ATP levels increase, the rate of this reaction decreases. If ATP is in short supply, the rate increases. If not already, the reason will become evident shortly.

Step 2: 
In step two, citrate loses one water molecule and gains another as citrate is converted into its isomer, isocitrate.

Step 3: 
In step three, isocitrate is oxidized by NAD+ and decarboxylated. Keep track of the carbons! This carbon now more than likely leaves the cell as waste and is no longer available for building new biomolecules. The oxidation of isocitrate therefore produces a five-carbon molecule, α-ketoglutarate, a molecule of CO2 and NADH. This step is also regulated by negative feedback from ATP and NADH, and via positive feedback from ADP.

Step 4: 
Step 4 is catalyzed by the enzyme succinate dehydrogenase. Here, α-ketoglutarate is further oxidized by NAD+. This oxidation again leads to a decarboxylation and thus the loss of another carbon as waste.  So far two carbons have come into the cycle from acetyl-CoA and two have left as CO2. At this stage, there is no net gain of carbons assimilated from the glucose molecules that are oxidized to this stage of metabolism. Unlike the previous step however succinate dehydrogenase - like pyruvate dehydrogenase before it - couples the free energy of the exergonic red/ox and decarboxylation reaction to drive the formation of a thioester bond between the substrate co-enzyme A and succinate (what is left after the decarboxylation). Succinate dehydrogenase is regulated by feedback inhibition of ATP, succinyl-CoA, and NADH.

SUGGESTED DISCUSSION

We have seen several steps in this and other pathways that are regulated by allosteric feedback mechanisms. Is there something(s) in common about these steps in the TCA cycle? Why might these be good steps to regulate?

SUGGESTED DISCUSSION 

The thioester bond has reappeared! Use the terms we've been learning (e.g. reduction, oxidation, coupling, exergonic, endergonic etc.) to describe the formation of this bond and below its hydrolysis.

 

Step 5: 
In step five, a substrate level phosphorylation event occurs. Here an inorganic phosphate (Pi) is added to GDP or ADP to form GTP (an ATP equivalent for our purposes) or ATP. The energy that drives this substrate level phosphorylation event comes from the hydrolysis of the CoA molecule from succinyl~CoA to form succinate. Why is either GTP or ATP produced? In animal cells there are two isoenzymes (different forms of an enzyme that carries out the same reaction), for this step, depending upon the type of animal tissue in which those cells are found. One isozyme is found in tissues that use large amounts of ATP, such as heart and skeletal muscle. This isozyme produces ATP. The second isozyme of the enzyme is found in tissues that have a large number of anabolic pathways, such as liver. This isozyme produces GTP. GTP is energetically equivalent to ATP; however, its use is more restricted. In particular, the process of protein synthesis primarily uses GTP. Most bacterial systems produce GTP in this reaction. 

Step 6: 
Step six is another red/ox reactions in which succinate is oxidized by FAD+ into fumarate. Two hydrogen atoms are transferred to FAD+, producing FADH2. The difference in reduction potential between the fumarate/succinate and NAD+/NADH half reactions is insufficient to make NAD+ a suitable reagent for oxidizing succinate with NAD+ under cellular conditions. However, the difference in reduction potential with the FAD+/FADH2 half reaction is adequate to oxidize succinate and reduce FAD+. Unlike NAD+, FAD+ remains attached to the enzyme and transfers electrons to the electron transport chain directly. This process is made possible by the localization of the enzyme catalyzing this step inside the inner membrane of the mitochondrion or plasma membrane (depending on whether the organism in question is eukaryotic or not).

Step 7:
Water is added to fumarate during step seven, and malate is produced. The last step in the citric acid cycle regenerates oxaloacetate by oxidizing malate with NAD+. Another molecule of NADH is produced in the process.

Summary
Note that this process (oxidation of pyruvate to Acetyl-CoA followed by one "turn" of the TCA cycle) completely oxidizes 1 molecule of pyruvate, a 3 carbon organic acid, to 3 molecules of CO2. Overall 4 molecules of NADH, 1 molecule of FADH2, and 1 molecule of GTP (or ATP) are also produced. For respiring organisms this is a significant mode of energy extraction, since each molecule of NADH and FAD2 can feed directly into the electron transport chain, and as we will soon see, the subsequent red/ox reactions that are driven by this process will indirectly power the synthesis of ATP. The discussion so far suggests that the TCA cycle is primarily an energy extracting pathway; evolved to extract or convert as much potential energy from organic molecules to a form that cells can use, ATP (or the equivalent) or an energized membrane. However, - and let us not forget - the other important outcome of evolving this pathway is the ability to produce several precursor or substrate molecules necessary for various catabolic reactions (this pathway provides some of the early building blocks to make bigger molecules). As we will discuss below, there is a strong link between carbon metabolism and energy metabolism.

EXERCISE

TCA Energy Stories

Work on building some energy stories yourself 

There are a few interesting reactions that involve large transfers of energy and rearrangements of matter. Pick a few. Rewrite a reaction in your notes, and practice constructing an energy story. You now have the tools to discuss the energy redistribution in the context of broad ideas and terms like exergonic and endergonic. You also have the ability to begin discussing mechanism (how these reactions happen) by invoking enzyme catalysts. See your instructor and/or TA and check with you classmates to self-test on how you're doing.

 

Connections to Carbon Flow
One hypothesis that we have started exploring in this reading and in class is the idea that "central metabolism" evolved as a means of generating carbon precursors for catabolic reactions. Our hypothesis also states that as cells evolved, these reactions became linked into pathways: glycolysis and the TCA cycle, as a means to maximize their effectiveness for the cell. We can postulate that a side benefit to evolving this metabolic pathway was the generation of NADH from the complete oxidation of glucose - we saw the beginning of this idea when we discussed fermentation. We have already discussed how glycolysis not only provides ATP from substrate level phosphorylation, but also yields a net of 2 NADH molecules and 6 essential precursors: glucose-6-P, fructose-6-P, 3-phosphoglycerate, phosphoenolpyruvate, and of course, pyruvate. While ATP can be used by the cell directly as an energy source, NADH posses a problem and must be recycled back into NAD+, to keep the pathway in balance. As we see in detail in the fermentation module, the most ancient way cells deal with this problem is to use fermentation reactions to regenerate NAD+. 

During the process of pyruvate oxidation via the TCA cycle 4 additional essential precursors are formed: acetyl~CoA, α-ketoglutarate, oxaloacetate, and succinyl~CoA. Three molecules of CO2 are lost and this represents a net loss of mass for the cell. These precursors, however, are substrates for a variety of catabolic reactions including the production of amino acids, fatty acids, and various co-factors, such as heme. This means that the rate of reactions through the TCA cycle will be sensitive to the concentrations of each metabolic intermediate (more on the thermodynamics in class). A metabolic intermediate is a compound that is produced by one reaction (a product) and then acts as a substrate for the next reaction. This also means that metabolic intermediates, in particular the 4 essential precursors, can be removed at any time for catabolic reactions, if there is a demand, changing the thermodynamics of the cycle. 

Not all cells have a functional TCA cycle
Since all cells require the ability of make these precursor molecules, one might expect that all organisms would have a fully functional TCA cycle. In fact, the cells of many organisms DO NOT have all of the enzymes required to form a complete cycle - all cells, however, DO have the capability of making the 4 TCA cycle precursors noted in the previous paragraph. How can the cells make precursors and not have a full cycle? Remember that most of these reactions are freely reversible, so, if NAD+ is required to for the oxidation of pyruvate or acetyl~CoA, then the reverse reactions would require NADH. This process is often referred to as the reductive TCA cycle. To drive these reactions in reverse (with respect to the direction discussed above) requires energy, in this case carried by ATP and NADH. If you get ATP and NADH driving a pathway one direction, it stands to reason that driving it in reverse will require ATP and NADH as "inputs".  So, organisms that do not have a full cycle can still make the 4 key metabolic precursors by using previously extracted energy and electrons (ATP and NADH) to drive some key steps in reverse.  

SUGGESTED DISCUSSION

Why might some organisms not have evolved a fully oxidative TCA cycle? Remember, cells need to keep a balance in the NAD+ to NADH ratio as well as the [ATP]/[AMP]/[ADP] ratios.

Additional Links
Here are some additional links to videos and pages that you may find useful.

Chemwiki Links
Chemwiki TCA cycle - link down until key content corrections are made to the resource
Khan Academy Links
Khan Academy TCA cycle - link down until key content corrections are made to the resource
 

 

 

Back to top
S2018_Lecture13_Reading  S2018_Lecture15_Reading
Recommended articles
S2018_Lecture01_Reading
S2018_Lecture02_Reading
S2018_Lecture03_Reading
S2018_Lecture04_Reading
S2018_Lecture05_Reading
The LibreTexts libraries are Powered by MindTouch® and are based upon work supported by the National Science Foundation under grant numbers: 1246120, 1525057, and 1413739. The California State University Affordable Learning Solutions and Merlot are the projects primary partners. Unless otherwise noted, the contents of the LibreTexts library is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 United States License. Permissions beyond the scope of this license may be available at delmarlarsen@gmail.com.
NSF Logo.png   imageedit_7_3300958659.png   imageedit_4_4211606159.png

Login to NB
'''
chapter_15 = '''Skip to main content
Help Spread the Word! The LibreTexts Project is the now the highest ranked and most visited online OER textbook project thanks to you. 
libretexts_section_complete_photon_124.png
Chemistry Biology Geosciences Mathematics Statistics Physics Social Sciences Engineering Medicine Business Photosciences Humanities
Search  
How can we help you?
 Search
 
Username
   
Password
   Sign in
 
Sign in
  Expand/collapse global hierarchy  Home   Course LibreTexts   University of California Davis   BIS 2A: Introductory Biology (Facciotti)   Readings   S2018_Lecture_Readings   Expand/collapse global location
S2018_Lecture15_Reading
Last updatedMay 6, 2018
S2018_Lecture14_Reading
 
S2018_Lecture16_Reading
picture_as_pdf
Donate
Introduction to Respiration and Electron Transport Chains
General Overview and Points to Keep In Mind

In the next few modules, we start to learn about the process of respiration and the roles that electron transport chains play in this process. A definition of the word "respiration" that most people are familiar with is "the act of breathing". When we breath, air including molecular oxygen is brought into our lungs from outside of the body, the oxygen then becomes reduced, and waste products, including the reduced oxygen in the form of water, are exhaled. More generically, some reactant comes into the organism and then gets reduced and leaves the body as a waste product.

This generic idea, in a nutshell, can be generally applied across biology. Note that oxygen need not always be the compound that brought in, reduced, and dumped as waste. The compounds onto which the electrons that are "dumped" are more specifically known as "terminal electron acceptors." The molecules from which the electrons originate vary greatly across biology (we have only looked at one possible source - the reduced carbon-based molecule glucose).  

In between the original electron source and the terminal electron acceptor are a series of biochemical reactions involving at least one red/ox reaction. These red/ox reactions harvest energy for the cell by coupling exergonic red/ox reaction to an energy-requiring reaction in the cell. In respiration, a special set of enzymes carry out a linked series of red/ox reactions that ultimately transfer electrons to the terminal electron acceptor.

These "chains" of red/ox enzymes and electron carriers are called electron transport chains (ETC). In aerobically respiring eukaryotic cells the ETC is composed of four large, multi-protein complexes embedded in the inner mitochondrial membrane and two small diffusible electron carriers shuttling electrons between them. The electrons are passed from enzyme to enzyme through a series of red/ox reactions. These reactions couple exergonic red/ox reactions to the endergonic transport of hydrogen ions across the inner mitochondrial membrane. This process contributes to the creation of a transmembrane electrochemical gradient. The electrons passing through the ETC gradually lose potential energy up until the point they are deposited on the terminal electron acceptor which is typically removed as waste from the cell. When oxygen acts as the final electron acceptor, the free energy difference of this multi-step red/ox process is ~ -60 kcal/mol when NADH donates electrons or ~ -45 kcal/mol when FADH2 donates.  

NOTE: OXYGEN IS NOT THE ONLY, NOR MOST FREQUENTLY USED, TERMINAL ELECTRON ACCEPTOR IN NATURE

Recall, that we use oxygen as an example of only one of numerous possible terminal electron acceptors that can be found in nature.  The free energy differences associated with respiration in anaerobic organisms will be different.

 

 

In prior modules we discussed the general concept of red/ox reactions in biology and introduced the Electron Tower, a tool to help you understand red/ox chemistry and to estimate the direction and magnitude of potential energy differences for various red/ox couples. In later modules, substrate level phosphorylation and fermentation were discussed and we saw how exergonic red/ox reactions could be directly coupled by enzymes to the endergonic synthesis of ATP.

These processes are hypothesized to be one of the oldest forms of energy production used by cells. In this section we discuss the next evolutionary advancement in cellular energy metabolism, oxidative phosphorylation. First and foremost recall that, oxidative phosphorylation does not imply the use of oxygen.  Rather the term oxidative phosphorylation is used because this process of ATP synthesis relies on red/ox reactions to generate a electrochemical transmembrane potential that can then be used by the cell to do the work of ATP synthesis. 

 

A Quick Overview of Principles Relevant to Electron Transport Chains
An ETC begins with the addition of electrons, donated from NADH, FADH2 or other reduced compounds. These electrons move through a series of electron transporters, enzymes that are embedded in a membrane, or other carriers that undergo red/ox reactions. The free energy transferred from these exergonic red/ox reactions is often coupled to the endergonic movement of protons across a membrane. Since the membrane is an effective barrier to charged species, this pumping results in an unequal accumulation of protons on either side of the membrane. This in turn "polarizes" or "charges" the membrane, with a net positive (protons) on one side of the membrane and a negative charge on the other side of the membrane. The separation of charge creates an electrical potential. In addition, the accumulation of protons also causes a pH gradient known as a chemical potential across the membrane. Together these two gradients (electrical and chemical) are called an electro-chemical gradient. 

Review: The Electron Tower
Since red/ox chemistry is so central to the topic we begin with a quick review of the table of reduction potential - sometimes called the "red/ox tower" or "electron tower". You may hear your instructors use these terms interchangeably. As we discussed in previous modules, all kinds of compounds can participate in biological red/ox reactions. Making sense of all of this information and ranking potential red/ox pairs can be confusing. A tool has been developed to rate red/ox half reactions based on their reduction potentials or E0' values. Whether a particular compound can act as an electron donor (reductant) or electron acceptor (oxidant) depends on what other compound it is interacting with. The red/ox tower ranks a variety of common compounds (their half reactions) from most negative E0', compounds that readily get rid of electrons, to the most positive E0', compounds most likely to accept electrons. The tower organizes these half reactions based on the ability of electrons to accept electrons. In addition, in many red/ox towers each half reaction is written by convention with the oxidized form on the left followed by the reduced form to its right. The two forms may be either separated by a slash, for example the half reaction for the reduction of NAD+ to NADH is written: NAD+/NADH + 2e-, or by separate columns. An electron tower is shown below.



 Figure 1. A common biological "red/ox tower"

NOTE

Use the red/ox tower above as a reference guide to orient you as to the reduction potential of the various compounds in the ETC. Red/ox reactions may be either exergonic or endergonic depending on the relative red/ox potentials of the donor and acceptor. Also remember there are many different ways of looking at this conceptually; this type of red/ox tower is just one way.

NOTE: LANGUAGE SHORTCUTS REAPPEAR

In the red/ox table above some entries seem to be written in unconventional ways. For instance Cytochrome cox/red. There only appears to be one form listed. Why? This is another example of language shortcuts (likely because someone was too lazy to write cytochrome twice) that can be confusing - particularly to students. The notation above could be rewritten as Cytochrome cox/Cytochrome cred to indicate that the cytochrome c protein can exist in either and oxidized state Cytochrome cox or reduced state Cytochrome cred.

Review Red/ox Tower Video
For a short video on how to use the red/ox tower in red/ox problems click here. This video was made by Dr. Easlon for Bis2A students.



 

Using the red/ox tower: A tool to help understand electron transport chains
By convention the tower half reactions are written with the oxidized form of the compound on the left and the reduced form on the right. Notice that compounds such as glucose and hydrogen gas are excellent electron donors and have very low reduction potentials E0'. Compounds, such as oxygen and nitrite, whose half reactions have relatively high positive reduction potentials (E0') generally make good electron acceptors are found at the opposite end of the table.

EXAMPLE: MENAQUINONE

Let's look at menaquinoneox/red. This compound sits in the middle of the red/ox tower with an half-reaction E0' value of -0.074 eV. Menaquinoneox can spontaneously (ΔG<0) accept electrons from reduced forms of compounds with lower half-reaction E0'. Such transfers form menaquinonered and the oxidized form of the original electron donor. In the table above, examples of compounds that could act as electron donors to menaquinone include FADH2, an E0' value of -0.22, or NADH, with an E0' value of -0.32 eV. Remember the reduced forms are on the right hand side of the red/ox pair. 

Once menaquinone has been reduced, it can now spontaneously (ΔG<0) donate electrons to any compound with a higher half-reaction E0' value. Possible electron acceptors include cytochrome box with an E0' value of 0.035 eV; or ubiquinoneox with an E0' of 0.11 eV. Remember that the oxidized forms lie on the left side of the half reaction.

 

 

Electron Transport Chains
An electron transport chain, or ETC, is composed of a group of protein complexes in and around a membrane that help energetically couple a series of exergonic/spontaneous red/ox reactions to the endergonic pumping of protons across the membrane to generate an electrochemical gradient. This electrochemical gradient creates a free energy potential that is termed a proton motive force whose energetically "downhill" exergonic flow can later be coupled to a variety of cellular processes.

ETC overview
Step 1: Electrons enter the ETC from an electron donor, such as NADH or FADH2, which are generated during a variety of catabolic reactions, including those associated glucose oxidation. Depending on the number and types of electron carriers of the ETC being used by an organism, electrons can enter at a variety of places in the electron transport chain. Entry of electrons at a specific "spot" in the ETC depends upon the respective reduction potentials of the electron donors and acceptors.


Step 2: After the first red/ox reaction, the initial electron donor will become oxidized and the electron acceptor will become reduced. The difference in red/ox potential between the electron acceptor and donor is related to ΔG by the relationship ΔG = -nFΔE, where n = the number of electrons transferred and F = Faraday's constant. The larger a positive ΔE, the more exergonic the red/ox reaction is.


Step 3: If sufficient energy is transferred during an exergonic red/ox step, the electron carrier may couple this negative change in free energy to the endergonic process of transporting a proton from one side of the membrane to the other.


Step 4: After usually multiple red/ox transfers, the electron is delivered to a molecule known as the terminal electron acceptor. In the case of humans, the terminal electron acceptor is oxygen. However, there are many, many, many, other possible electron acceptors in nature; see below.

 

NOTE: POSSIBLE DISCUSSION

Electrons entering the ETC do not have to come from NADH or FADH2. Many other compounds can serve as electron donors; the only requirements are (1) that there exists an enzyme that can oxidize the electron donor and then reduce another compound, and (2) that the ∆E0' is positive (e.g., ΔG<0). Even a small amounts of free energy transfers can add up. For example, there are bacteria that use H2 as an electron donor. This is not too difficult to believe because the half reaction 2H+ + 2 e-/H2 has a reduction potential (E0') of -0.42 V. If these electrons are eventually delivered to oxygen, then the ΔE0' of the reaction is 1.24 V, which corresponds to a large negative ΔG (-ΔG). Alternatively, there are some bacteria that can oxidize iron, Fe2+ at pH 7 to Fe3+ with a reduction potential (E0') of + 0.2 V. These bacteria use oxygen as their terminal electron acceptor, and, in this case, the ΔE0' of the reaction is approximately 0.62 V. This still produces a -ΔG. The bottom line is that, depending on the electron donor and acceptor that the organism uses, a little or a lot of energy can be transferred and used by the cell per electrons donated to the electron transport chain.

What are the complexes of the ETC?
ETCs are made up of a series (at least one) of membrane-associated red/ox proteins or (some are integral) protein complexes (complex = more than one protein arranged in a quaternary structure) that move electrons from a donor source, such as NADH, to a final terminal electron acceptor, such as oxygen. This particular donor/terminal acceptor pair is the primary one used in human mitochondria. Each electron transfer in the ETC requires a reduced substrate as an electron donor and an oxidized substrate as the electron acceptor. In most cases, the electron acceptor is a member of the enzyme complex itsef. Once the complex is reduced, the complex can serve as an electron donor for the next reaction.

How do ETC complexes transfer electrons?
As previously mentioned, the ETC is composed of a series of protein complexes that undergo a series of linked red/ox reactions. These complexes are in fact multi-protein enzyme complexes referred to as oxidoreductases or simply, reductases. The one exception to this naming convention is the terminal complex in aerobic respiration that uses molecular oxygen as the terminal electron acceptor. That enzyme complex is referred to as an oxidase. Red/ox reactions in these complexes are typically carried out by a non-protein moiety called a prosthetic group. The prosthetic groups are directly involved in the red/ox reactions being catalyzed by their associated oxidoreductases. In general, these prosthetic groups can be divided into two general types: those that carry both electrons and protons and those that only carry electrons.

NOTE

This use of prosthetic groups by members of ETC is true for all of the electron carriers with the exception of quinones, which are a class of lipids that can directly be reduced or oxidized by the oxidoreductases. Both the Quinone(red) and the Quinone(ox) forms of these lipids are soluble within the membrane and can move from complex to complex to shuttle electrons.

The electron and proton carriers
Flavoproteins (Fp), these proteins contain an organic prosthetic group called a flavin, which is the actual moiety that undergoes the oxidation/reduction reaction. FADH2 is an example of an Fp.
Quinones are a family of lipids, which means they are soluble within the membrane.
It should also be noted that NADH and NADPH are considered electron (2e-) and proton (2 H+) carriers.
Electron carriers
Cytochromes are proteins that contain a heme prosthetic group. The heme is capable of carrying a single electron.
Iron-Sulfur proteins contain a nonheme iron-sulfur cluster that can carry an electron. The prosthetic group is often abbreviated as Fe-S
Aerobic versus anaerobic respiration
We humans use oxygen as the terminal electron acceptor for the ETCs in our cells.  This is also the case for many of the organisms we intentionally and frequently interact with (e.g. our classmates, pets, food animals, etc). We breath in oxygen; our cells take it up and transport it into the mitochondria where it is used as the final acceptor of electrons from our electron transport chains. That process - because oxygen is used as the terminal electron acceptor - is called aerobic respiration.

While we may use oxygen as the terminal electron acceptor for our respiratory chains, this is not the only mode of respiration on the planet.  Indeed, the more general processes of respiration evolved at a time when oxygen was not a major component of the atmosphere. As a consequence, many organisms can use a variety of compounds including nitrate (NO3-), nitrite (NO2-), even iron (Fe3+) as terminal electron acceptors. When oxygen is NOT the terminal electron acceptor, the process is referred to as anaerobic respiration. Therefore, respiration or oxidative phosphorylation does not require oxygen at all; it simply requires a compound with a high enough reduction potential to act as a terminal electron acceptor, accepting electrons from one of the complexes within the ETC.

The ability of some organisms to vary their terminal electron acceptor provides metabolic flexibility and can ensure better survival if any given terminal acceptor is in limited supply. Think about this: in the absence of oxygen, we die; but other organisms can use a different terminal electron acceptor when conditions change in order to survive.

A generic example: A simple, two-complex ETC
The figure below depicts a generic electron transport chain, composed of two integral membrane complexes; Complex I(ox) and Complex II(ox). A reduced electron donor, designated DH (such as NADH or FADH2) reduces Complex I(ox), giving rise to the oxidized form D (such as NAD+ or FAD+). Simultaneously, a prosthetic group within Complex I is now reduced (accepts the electrons). In this example, the red/ox reaction is exergonic and the free energy difference is coupled by the enzymes in Complex I to the endergonic translocation of a proton from one side of the membrane to the other. The net result is that one surface of the membrane becomes more negatively charged, due to an excess of hydroxyl ions (OH-), and the other side becomes positively charged due to an increase in protons on the other side. Complex I(red) can now reduce a mobile electron carrier Q, which will then move through the membrane and transfer the electron(s) to the prosthetic group of Complex II(red). Electrons pass from Complex I to Q then from Q to Complex II via thermodynamically spontaneous red/ox reactions, regenerating Complex I(ox), which can repeat the previous process. Complex II(red) then reduces A, the terminal electron acceptor to regenerate Complex II(ox) and create the reduced form of the terminal electron acceptor, AH. In this specific example, Complex II can also translocate a proton during the process. If A is molecular oxygen, AH represents water and the process would be considered to be a model of an aerobic ETC.  By contrast, if A is nitrate, NO3-, then AH represents NO2- (nitrite) and this would be an example of an anaerobic ETC.

 



 

Figure 1. Generic 2 complex electron transport chain. In the figure, DH is the electron donor (donor reduced), and D is the donor oxidized. A is the oxidized terminal electron acceptor, and AH is the final product, the reduced form of the acceptor. As DH is oxidized to D, protons are translocated across the membrane, leaving an excess of hydroxyl ions (negatively charged) on one side of the membrane and protons (positively charged) on the other side of the membrane. The same reaction occurs in Complex II as the terminal electron acceptor is reduced to AH.

Attribution: Marc T. Facciotti (original work)

 

EXERCISE 1

Thought question

Based on the figure above, use an electron tower to figure out the difference in the electrical potential if (a) DH is NADH and A is O2, and (b) DH is NADH and A is NO3-. Which pairs of electron donor and terminal electron acceptor (a) or (b) "extract" the greatest amount of free energy?

Detailed look at aerobic respiration
The eukaryotic mitochondria has evolved a very efficient ETC. There are four complexes composed of proteins, labeled I through IV depicted in the figure below. The aggregation of these four complexes, together with associated mobile, accessory electron carriers, is called also called an electron transport chain. This type of electron transport chain is present in multiple copies in the inner mitochondrial membrane of eukaryotes.



Figure 2. The electron transport chain is a series of electron transporters embedded in the inner mitochondrial membrane that shuttles electrons from NADH and FADH2 to molecular oxygen. In the process, protons are pumped from the mitochondrial matrix to the intermembrane space, and oxygen is reduced to form water.

Complex I
To start, two electrons are carried to the first protein complex aboard NADH. This complex, labeled I in Figure 2, includes flavin mononucleotide (FMN) and iron-sulfur (Fe-S)-containing proteins. FMN, which is derived from vitamin B2, also called riboflavin, is one of several prosthetic groups or cofactors in the electron transport chain. Prosthetic groups are organic or inorganic, nonpeptide molecules bound to a protein that facilitate its function; prosthetic groups include coenzymes, which are the prosthetic groups of enzymes. The enzyme in Complex I is also called NADH dehydrogenase and is a very large protein containing 45 individual polypeptide chains. Complex I can pump four hydrogen ions across the membrane from the matrix into the intermembrane space thereby helping to generate and maintain a hydrogen ion gradient between the two compartments separated by the inner mitochondrial membrane.

Q and Complex II
Complex II directly receives FADH2, which does not pass through Complex I. The compound connecting the first and second complexes to the third is ubiquinone (Q). The Q molecule is lipid soluble and freely moves through the hydrophobic core of the membrane. Once it is reduced, (QH2), ubiquinone delivers its electrons to the next complex in the electron transport chain. Q receives the electrons derived from NADH from Complex I and the electrons derived from FADH2 from Complex II, succinate dehydrogenase. Since these electrons bypass and thus do not energize the proton pump in the first complex, fewer ATP molecules are made from the FADH2 electrons. As we will see in the following section, the number of ATP molecules ultimately obtained is directly proportional to the number of protons pumped across the inner mitochondrial membrane.

Complex III
The third complex is composed of cytochrome b, another Fe-S protein, Rieske center (2Fe-2S center), and cytochrome c proteins; this complex is also called cytochrome oxidoreductase. Cytochrome proteins have a prosthetic group of heme. The heme molecule is similar to the heme in hemoglobin, but it carries electrons, not oxygen. As a result, the iron ion at its core is reduced and oxidized as it passes the electrons, fluctuating between different oxidation states: Fe2+ (reduced) and Fe3+ (oxidized). The heme molecules in the cytochromes have slightly different characteristics due to the effects of the different proteins binding them, giving slightly different characteristics to each complex. Complex III pumps protons through the membrane and passes its electrons to cytochrome c for transport to the fourth complex of proteins and enzymes (cytochrome c is the acceptor of electrons from Q; however, whereas Q carries pairs of electrons, cytochrome c can accept only one at a time).

Complex IV
The fourth complex is composed of cytochrome proteins c, a, and a3. This complex contains two heme groups (one in each of the two Cytochromes, a, and a3) and three copper ions (a pair of CuA and one CuB in Cytochrome a3). The cytochromes hold an oxygen molecule very tightly between the iron and copper ions until the oxygen is completely reduced. The reduced oxygen then picks up two hydrogen ions from the surrounding medium to make water (H2O). The removal of the hydrogen ions from the system contributes to the ion gradient used in the process of chemiosmosis.

Chemiosmosis
In chemiosmosis, the free energy from the series of red/ox reactions just described is used to pump protons across the membrane. The uneven distribution of H+ ions across the membrane establishes both concentration and electrical gradients (thus, an electrochemical gradient), owing to the proton's positive charge and their aggregation on one side of the membrane.

If the membrane were open to diffusion by protons, the ions would tend to diffuse back across into the matrix, driven by their electrochemical gradient. Ions, however, cannot diffuse through the nonpolar regions of phospholipid membranes without the aid of ion channels. Similarly, protons in the intermembrane space can only traverse the inner mitochondrial membrane through an integral membrane protein called ATP synthase (depicted below). This complex protein acts as a tiny generator, turned by transfer of energy mediated by protons moving down their electrochemical gradient. The movement of this molecular machine (enzyme) serves to lower the activation energy of reaction and couples the exergonic transfer of energy associated with the movement of protons down their electrochemical gradient to the endergonic addition of a phosphate to ADP, forming ATP.



 Figure 3. ATP synthase is a complex, molecular machine that uses a proton (H+) gradient to form ATP from ADP and inorganic phosphate (Pi).

Credit: modification of work by Klaus Hoffmeier

NOTE: POSSIBLE DISCUSSION

Dinitrophenol (DNP) is a small chemical that serves to uncouple the flow of protons across the inner mitochondrial membrane to the ATP synthase, and thus the synthesis of ATP. DNP makes the membrane leaky to protons. It was used until 1938 as a weight-loss drug. What effect would you expect DNP to have on the difference in pH across both sides of the inner mitochondrial membrane? Why do you think this might be an effective weight-loss drug? Why might it be dangerous?

In healthy cells, chemiosmosis (depicted below) is used to generate 90 percent of the ATP made during aerobic glucose catabolism; it is also the method used in the light reactions of photosynthesis to harness the energy of sunlight in the process of photophosphorylation. Recall that the production of ATP using the process of chemiosmosis in mitochondria is called oxidative phosphorylation and that a similar process can occur in the membranes of bacterial and archaeal cells. The overall result of these reactions is the production of ATP from the energy of the electrons removed originally from a reduced organic molecule like glucose. In the aerobic example, these electrons ultimatel reduce oxygen and thereby create water.

 



Figure 4. In oxidative phosphorylation, the pH gradient formed by the electron transport chain is used by ATP synthase to form ATP in a Gram-bacteria.

 

Helpful link: How ATP is made from ATP synthase



 

NOTE: POSSIBLE DISCUSSION

Cyanide inhibits cytochrome c oxidase, a component of the electron transport chain. If cyanide poisoning occurs, would you expect the pH of the intermembrane space to increase or decrease? What effect would cyanide have on ATP synthesis?

 

A Hypothesis for How ETC May Have Evolved
A proposed link between SLP/fermentation and the evolution of ETCs:

In a previous discussion of energy metabolism, we explored substrate level phosphorylation (SLP) and fermentation reactions. One of the questions in the discussion points for that discussion was: what are the short- and long-term consequences of SLP to the environment? We discussed how cells needed to co-evolve mechanisms in order to remove protons from the cytosol (interior of the cell), which led to the evolution of the F0F1-ATPase, a multi-subunit enzyme that translocates protons from the inside of the cell to the outside of the cell by hydrolyzing ATP, as shown below in the first picture below. This arrangement works as long as small reduced organic molecules are freely available, making SLP and fermentation advantageous. However, as these biological processes continue, the small reduced organic molecules begin to be used up, and their concentration decreases; this puts a demand on cells to be more efficient.

One source of potential "ATP waste" is in the removal of protons from the cell's cytosol; organisms that could find other mechanisms to  expel accumulating protons while still preserving ATP could have a selective advantage. It is hypothesized that this selective evolutionary pressure potentially led to the evolution of the first membrane-bound proteins that used red/ox reactions as their energy source (depicted in second picture) to pump the accumulating protons. Enzymes and enzyme complexes with these properties exist today in the form of the electron transport complexes like Complex I, the NADH dehydrogenase.



Figure 1. Proposed evolution of an ATP dependent proton translocator



Figure 2. As small reduced organic molecules become limited, organisms that can find alternative mechanisms to remove protons from the cytosol may have an advantage. The evolution of a proton translocator that uses red/ox reactions rather than ATP hydrolysis could substitute for the ATPase.

Continuing with this line of logic, if organisms evolved that could now use red/ox reactions to translocate protons across the membrane they would create a an electrochemical gradient, separating both charge (positive on the outside and negative on the inside; an electrical potential) and pH (low pH outside, higher pH inside). With excess protons on the outside of the cell membrane, and the F0F1-ATPase no longer consuming ATP to translocate protons, it is hypothesized that the electrochemical gradient could then be used to power the F0F1-ATPase "backwards" — that is, to form or produce ATP by using the energy in the charge/pH gradients set up by the red/ox pumps (as depicted below). This arrangement is called an electron transport chain (ETC).



Figure 3. The evolution of the ETC; the combination of the red/ox driven proton translocators coupled to the production of ATP by the F0F1-ATPase.

 

NOTE: EXTENDED READING ON THE EVOLUTION OF ELECTRON TRANSPORT CHAINS

If you're interested in the story of the evolution of electron transport chains, check out this more in-depth discussion of the topic at NCBI.  

 

 

Back to top
S2018_Lecture14_Reading  S2018_Lecture16_Reading
Recommended articles
S2018_Lecture01_Reading
S2018_Lecture02_Reading
S2018_Lecture03_Reading
S2018_Lecture04_Reading
S2018_Lecture05_Reading
The LibreTexts libraries are Powered by MindTouch® and are based upon work supported by the National Science Foundation under grant numbers: 1246120, 1525057, and 1413739. The California State University Affordable Learning Solutions and Merlot are the projects primary partners. Unless otherwise noted, the contents of the LibreTexts library is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 United States License. Permissions beyond the scope of this license may be available at delmarlarsen@gmail.com.
NSF Logo.png   imageedit_7_3300958659.png   imageedit_4_4211606159.png

Login to NB
'''
chapter_16= '''Skip to main content
Help Spread the Word! The LibreTexts Project is the now the highest ranked and most visited online OER textbook project thanks to you. 
libretexts_section_complete_photon_124.png
Chemistry Biology Geosciences Mathematics Statistics Physics Social Sciences Engineering Medicine Business Photosciences Humanities
Search  
How can we help you?
 Search
 
Username
   
Password
   Sign in
 
Sign in
  Expand/collapse global hierarchy  Home   Course LibreTexts   University of California Davis   BIS 2A: Introductory Biology (Facciotti)   Readings   S2018_Lecture_Readings   Expand/collapse global location
S2018_Lecture16_Reading
Last updatedMay 6, 2018
S2018_Lecture15_Reading
 
S2018_Lecture17_Reading
picture_as_pdf
Donate
Light Energy and Pigments
Light Energy
The sun emits an enormous amount of electromagnetic radiation (solar energy) that spans a broad swath of the electromagnetic spectrum, the range of all possible radiation frequencies. When solar radiation reaches Earth, a fraction of this energy interacts with and may be transferred to the matter on the planet. This energy transfer results in a wide variety of different phenomena, from influencing weather patterns to driving a myriad of biological processes. In BIS2A, we are largely concerned with the latter, and below, we discuss some very basic concepts related to light and its interaction with biology.  

First, however we need to refresh a couple of key properties of light: 

Light in a vacuum travels at a constant speed of 299,792,458 m/s. We often abbreviate the speed of light with the variable "c".  
Light has properties of waves. A specific "color" of light has a characteristic wavelength.


The distance between peaks in a wave is referred to as the wavelength and is abbreviated with the greek letter lambda (Ⲗ).  

Attribution: Marc T. Facciotti (original work)

 



The inverse proportionality of frequency and wavelength. Wave 1 has a wavelength that is 2x that of wave 2 (Ⲗ1 > Ⲗ2). If the two waves are traveling at the same speed (c)—imagine that both of the whole lines that are drawn are dragged past the fixed vertical line at the same speed —then the number of times a wave peak passes a fixed point is greater for wave 2 than wave 1 (f2 > f1).  

Attribution: Marc T. Facciotti (original work)

3. Finally, each frequency (or wavelength) of light is associated with a specific energy. We'll call energy "E". The relationship between frequency and energy is:

E = h*f

where h is a constant called the Planck constant (~6.626x10-34 Joule•second when frequency is expressed in cycles per second). Given the relationship between frequency and wavelength, you can also write E = h*c/Ⲗ. Therefore, the larger the frequency (or shorter the wavelength), the more energy is associated with a specific "color". Wave 2 in the figure above is associated with greater energy than wave 1.



The sun emits energy in the form of electromagnetic radiation. All electromagnetic radiation, including visible light, is characterized by its wavelength. The longer the wavelength, the less energy it carries. The shorter the wavelength, the more energy is associated with that band of the electromagnetic spectrum.

The Light We See
The visible light seen by humans as white light is composed of a rainbow of colors, each with a characteristic wavelength. Certain objects, such as a prism or a drop of water, disperse white light to reveal the colors to the human eye. In the visible spectrum, violet and blue light have shorter (higher energy) wavelengths while the orange and red light have longer (lower energy) wavelengths.



The colors of visible light do not carry the same amount of energy. Violet has the shortest wavelength and therefore carries the most energy, whereas red has the longest wavelength and carries the least amount of energy.

Credit: modification of work by NASA

Absorption by Pigments
The interaction between light and biological systems occurs through several different mechanisms, some of which you may learn about in upper division courses in cellular physiology or biophysical chemistry. In BIS2A, we are mostly concerned with the interaction of light and biological pigments. These interactions can initiate a variety of light-dependent biological processes that can be grossly grouped into two functional categories: cellular signaling and energy harvesting. Signaling interactions are largely responsible for perceiving changes in the environment (in this case, changes in light). An example of a signaling interaction might be the interaction between light and the pigments expressed in an eye. By contrast, light/pigment interactions that are involved in energy harvesting are used for—not surprisingly—capturing the energy in the light and transferring it to the cell to fuel biological processes. Photosynthesis, which we will learn more about soon, is one example of an energy harvesting interaction.  

At the center of the biological interactions with light are groups of molecules we call organic pigments. Whether in the human retina, chloroplast thylakoid, or microbial membrane, organic pigments often have specific ranges of energy or wavelengths that they can absorb. The sensitivity of these molecules for different wavelengths of light is due to their unique chemical makeups and structures. A range of the electromagnetic spectrum is given a couple of special names because of the sensitivity of some key biological pigments: The retinal pigment in our eyes, when coupled with an opsin sensor protein, “sees” (absorbs) light predominantly between the wavelengths between of 700 nm and 400 nm. Because this range defines the physical limits of the electromagnetic spectrum that we can actually see with our eyes, we refer to this wavelength range as the "visible range". For similar reasons, as plants pigment molecules tend to absorb wavelengths of light mostly between 700 nm and 400 nm, plant physiologists refer to this range of wavelengths as "photosynthetically active radiation".

Three Key Types of Pigments We Discuss in BIS2A
Chlorophylls 
Chlorophylls (including bacteriochlorophylls) are part of a large family of pigment molecules. There are five major chlorophyll pigments named: a, b, c, d, and f. Chlorophyll a is related to a class of more ancient molecules found in bacteria called bacteriochlorophylls. Chlorophylls are structurally characterized by ring-like porphyrin group that coordinates a metal ion. This ring structure is chemically related to the structure of heme compounds that also coordinate a metal and are involved in oxygen binding and/or transport in many organisms. Different chlorophylls are distinguished from one another by different "decorations"/chemical groups on the porphyrin ring.



The structure of heme and chlorophyll a molecules. The common porphyrin ring is colored in red.

Attribution: Marc T. Facciotti (original work)

Carotenoids 
Carotenoids are the red/orange/yellow pigments found in nature. They are found in fruit—the red of tomato (lycopene), the yellow of corn seeds (zeaxanthin), or the orange of an orange peel (β-carotene)—which are used as biological "advertisements" to attract seed dispersers (animals or insects that may carry seeds elsewhere). In photosynthesis, carotenoids function as photosynthetic pigments. In addition, when a leaf is exposed to full sun, that surface is required to process an enormous amount of energy; if that energy is not handled properly, it can do significant damage. Therefore, many carotenoids help absorb excess energy in light and safely dissipate that energy as heat. 

 

Flavonoids  
Flavonoids are a very broad class of compounds that are found in great diversity in plants.  These molecules come in many forms but all share a common core structure shown below. The diversity of flavonoids comes from the many different combinations of functional groups that can "decorate" the core flavone.  

 



The core ring structure of flavans.  

 

Each type of pigment can be identified by the specific pattern of wavelengths it absorbs from visible light. This characteristic is known as the pigment's absorption spectrum. The graph in the figure below shows the absorption spectra for chlorophyll a, chlorophyll b, and a type of carotenoid pigment called β-carotene (which absorbs blue and green light). Notice how each pigment has a distinct set of peaks and troughs, revealing a highly specific pattern of absorption. These differences in absorbance are due to differences in chemical structure (some are highlighted in the figure). Chlorophyll a absorbs wavelengths from either end of the visible spectrum (blue and red), but not green. Because green is reflected or transmitted, chlorophyll appears green. Carotenoids absorb in the short-wavelength blue region, and reflect the longer yellow, red, and orange wavelengths.



(a) Chlorophyll a, (b) chlorophyll b, and (c) β-carotene are hydrophobic organic pigments found in the thylakoid membrane. Chlorophyll a and b, which are identical except for the part indicated in the red box, are responsible for the green color of leaves. Note how the small amount of difference in chemical composition between different chlorophylls leads to different absorption spectra. β-carotene is responsible for the orange color in carrots. Each pigment has a unique absorbance spectrum (d).

IMPORTANCE OF HAVING MULTIPLE DIFFERENT PIGMENTS

Not all photosynthetic organisms have full access to sunlight. Some organisms grow underwater where light intensity and available wavelengths decrease and change, respectively, with depth. Other organisms grow in competition for light. For instance, plants on the rainforest floor must be able to absorb any bit of light that comes through because the taller trees absorb most of the sunlight and scatter the remaining solar radiation. To account for these variable light conditions, many photosynthetic organisms have a mixture of pigments whose expression can be tuned to improve the organism's ability to absorb energy from a wider range of wavelengths than would be possible with one pigment alone. 

 

Photophosphorylation
Photophosphorylation an overview
Photophosphorylation is the process of transferring the energy from light into chemicals, particularly ATP. The evolutionary roots of photophosphorylation are likely in the anaerobic world, between 3 billion and 1.5 billion years ago, when life was abundant in the absence of molecular oxygen. Photophosphorylation probably evolved relatively shortly after electron transport chains (ETC) and anaerobic respiration began to provide metabolic diversity. The first step of the process involves the absorption of a photon by a pigment molecule. Light energy is transferred to the pigment and promotes electrons (e-) into a higher quantum energy state—something biologists term an "excited state". Note the use of anthropomorphism here; the electrons are not "excited" in the classic sense and aren't all of a sudden hopping all over or celebrating their promotion. They are simply in a higher energy quantum state. In this state, the electrons are colloquially said to be "energized". While in the "excited" state, the pigment now has a much lower reduction potential and can donate the "excited" electrons to other carriers with greater reduction potentials. These electron acceptors may, in turn, become donors to other molecules with greater reduction potentials and, in doing so, form an electron transport chain. 

As electrons pass from one electron carrier to another via red/ox reactions, these exergonic transfers can be coupled to the endergonic transport (or pumping) of protons across a membrane to create an electrochemical gradient. This electrochemical gradient generates a proton motive force whose exergonic drive to reach equilibrium can be coupled to the endergonic production of ATP, via ATP synthase. As we will see in more detail, the electrons involved in this electron transport chain can have one of two fates: (1) they may be returned to their initial source in a process called cyclic photophosphorylation; or (2) they can be deposited onto a close relative of NAD+ called NADP+. If the electrons are deposited back on the original pigment in a cyclic process, the whole process can start over. If, however, the electron is deposited onto NADP+ to form NADPH (**shortcut note—we didn't explicitly mention any protons but assume it is understood that they are also involved**), the original pigment must regain an electron from somewhere else. This electron must come from a source with a smaller reduction potential than the oxidized pigment and depending on the system there are different possible sources, including H2O, reduced sulfur compounds such as SH2 and even elemental S0.

What happens when a compound absorbs a photon of light?
When a compound absorbs a photon of light, the compound is said to leave its ground state and become "excited".



Figure 1. A diagram depicting what happens to a molecule that absorbs a photon of light. Attribution: Marc T. Facciotti (original work)

What are the fates of the "excited" electron? There are four possible outcomes, which are schematically diagrammed in the figure below. These options are:

The e- can relax to a lower quantum state, transferring energy as heat. 
The e- can relax to a lower quantum state and transfer energy into a photon of light—a process known as fluorescence. 
The energy can be transferred by resonance to a neighboring molecule as the e- returns to a lower quantum state. 
The energy can change the reduction potential such that the molecule can become an e- donor. Linking this excited e- donor to a proper e- acceptor can lead to an exergonic electron transfer. In other words, the excited state can be involved in red/ox reactions.


Figure 2. What can happen to the energy absorbed by a molecule.
As the excited electron decays back to its lower energy state, the energy can be transferred in a variety of ways. While many so called antenna or auxiliary pigments absorb light energy and transfer it to something known as a reaction center (by mechanisms depicted in option III in Figure 2), it is what happens at the reaction center that we are most concerned with (option IV in the figure above). Here a chlorophyll or bacteriochlorophyll molecule absorbs a photon's energy and an electron is excited. This energy transfer is sufficient to allow the reaction center to donate the electron in a red/ox reaction to a second molecule. This initiates the electron transport reactions. The result is an oxidized reaction center that must now be reduced in order to start the process again. How this happens is the basis of electron flow in photophosphorylation and will be described in detail below. 

Simple photophosphorylation systems: anoxygenic photophosphorylation
Early in the evolution of photophosphorylation, these reactions evolved in anaerobic environments where there was very little molecular oxygen available. Two sets of reactions evolved under these conditions, both directly from anaerobic respiratory chains as described previously. These are known as the light reactions because they require the activation of an electron (an "excited" electron) from the absorption of a photon of light by a reaction center pigment, such as bacteriochlorophyll. The light reactions are categorized either as cyclic or as noncyclic photophosphorylation, depending upon the final state of the electron(s) removed from the reaction center pigments. If the electron(s) returns to the original pigment reaction center, such as bacteriochlorophyll, this is cyclic photophosphorylation; the electrons make a complete circuit and is diagramed in Figure 4. If the electron(s) are used to reduce NADP+ to NADPH, the electron(s) are removed from the pathway and end up on NADPH; this process is referred to as noncyclic since the electrons are no longer part of the circuit. In this case the reaction center must be re-reduced before the process can happen again. Therefore, an external electron source is required for noncylic photophosphorylation. In these systems reduced forms of Sulfur, such as H2S, which can be used as an electron donor and is diagrammed in Figure 5. To help you better understand the similarities of photophosphorylation to respiration, a red/ox tower has been provided that contains many commonly used compounds involved with photosphosphorylation.

 

 

 oxidized form

reduced form

n (electrons)

Eo´ (volts)

PS1* (ox)

PS1* (red)

-

-1.20

ferredoxin (ox) version 1

ferredoxin (red) version 1

1

-0.7

PSII* (ox)

PSII* (red)

-

-0.67

P840* (ox)

PS840* (red)

-

-0.67

acetate

acetaldehyde

2

-0.6

CO2

Glucose

24

-0.43

ferredoxin (ox) version 2

ferredoxin (red) version 2

1

-0.43

CO2 

formate

2

-0.42

2H+

H2

2

-0.42

NAD+ + 2H+

NADH  + H+

2

-0.32

NADP+ + 2H+

NADPH  + H+

2

-0.32

Complex I

FMN (enzyme bound)

FMNH2

2

-0.3

Lipoic acid, (ox)

Lipoic acid, (red)

2

-0.29

FAD+ (free) + 2H+

FADH2

2

-0.22

Pyruvate + 2H+

lactate

2

-0.19

FAD+ + 2H+ (bound)

FADH2 (bound)

2

0.003-0.09

CoQ (Ubiquinone - UQ + H+)

UQH.

1

0.031

UQ + 2H+

UQH2

2

0.06

Plastoquinone; (ox)

Plastoquinone; (red)

-

0.08

Ubiquinone; (ox)

Ubiquinone; (red)

2

0.1

Complex III Cytochrome b2; Fe3+

Cytochrome b2; Fe2+

1

0.12

Complex III Cytochrome c1; Fe3+

Cytochrome c1; Fe2+

1

0.22

Cytochrome c; Fe3+

Cytochrome c; Fe2+

1

0.25

Complex IV Cytochrome a; Fe3+

Cytochrome a; Fe2+

1

0.29

1/2 O2 + H2O

H2O2

2

0.3

P840GS (ox)

PS840GS (red)

-

0.33

Complex IV Cytochrome a3; Fe3+

Cytochrome a3; Fe2+

1

0.35

Ferricyanide

ferrocyanide

2

0.36

Cytochrome f; Fe3+

Cytochrome f; Fe2+

1

0.37

PSIGS (ox)

PSIGS (red)

.

0.37

Nitrate

nitrite

1

0.42

Fe3+

Fe2+

1

0.77

1/2 O2 + 2H+

H2O

2

0.816

PSIIGS (ox)

PSIIGS (red)

-

1.10

* Excited State, after absorbing a photon of light

GS Ground State, state prior to absorbing a photon of light

PS1: Oxygenic photosystem I

P840: Bacterial reaction center containing bacteriochlorophyll (anoxygenic)

PSII: Oxygenic photosystem II 

Figure 3. Electron tower that has a variety of common photophosphorylation components. PSI and PSII refer to Photosystems I and II of the oxygenic photophosphorylation pathways.
Cyclic photophosphorylation
In cyclic photophosphorylation the bacteriochlorophyllred molecule absorbs enough light energy to energize and eject an electron to form bacteriochlorophyllox. The electron reduces a carrier molecule in the reaction center which in turn reduces a series of carriers via red/ox reactions. These carriers are the same carriers found in respiration. If the change in reduction potential from the various red/ox reactions are sufficiently large, H+ protons can be translocated across a membrane. Eventually, the electron is used to reduce bacteriochlorophyllox (making a complete loop) and the whole process can start again. This flow of electrons is cyclic and is therefore said to drive a processed called cyclic photophosphorylation. The electrons make a complete cycle: bacteriochlorophyll is the initial source of electrons and is the final electron acceptor. ATP is produced via the F1F0 ATPase. The schematic in Figure 4 demonstrates how cyclic electrons flow and thus how cyclic photophosphorylation works. 

 



Figure 4. Cyclic electron flow. The reaction center P840 absorbs light energy and becomes excited, denoted with an *. The excited electron is ejected and used to reduce an FeS protein leaving an oxidized reaction center. The electron its transferred to a quinone, then to a series of cytochromes, which in turn reduces the P840 reaction center. The process is cyclical. Note the gray array coming from the FeS protein going to a ferridoxin (Fd), also in gray. This represents an alternative pathway the electron can take and will be discussed below in noncyclic photophosphorylation. Note that the electron that initially leaves the P840 reaction center is not necessarily the same electron that eventually finds its way back to reduce the oxidized P840. 

NOTE: POSSIBLE DISCUSSION

The figure of cyclic photophosphorylation above depicts the flow of electrons in a respiratory chain. How does this process help generate ATP?

Noncyclic photophosphorylation
In cyclic photophosphorylation, electrons cycle from bacteriochlorophyll (or chlorophyll) to a series of electron carriers and eventually back to bacteriochlorophyll (or chlorophyll); there is theoretically no net loss of electrons and they stay in the system. In noncyclic photophosphorylation, electrons are removed from the photosystem and red/ox chain and eventually end up on NADPH. That means there needs to be a source of electrons, a source that has a smaller reduction potential than bacteriochlorophyll (or chlorophyll) that can donate electrons to bacteriochlorophyllox to reduce it. From looking at the electron tower in Figure 3, you can see what compounds can be used to reduce the oxidized form of bacteriochlorophyll. The second requirement is that, when bacteriochlorophyll becomes oxidized and the electron is ejected, it must reduce a carrier that has a greater reduction potential than NADP/NADPH (see the electron tower). In this case, electrons can flow from energized bacteriochlorophyll to NADP forming NADPH and oxidized bacteriochlorophyll. Electrons are lost from the system and end up on NADPH; to complete the circuit, bacteriochlorophyllox is reduced by an external electron donor such as H2S or elemental S0.

Noncyclic electron flow


Figure 5. Noncyclic electron flow. In this example, the P840 reaction center absorbs light energy and becomes energized; the emitted electron reduces a FeS protein and in turn reduces ferridoxin. Reduced ferridoxin (Fdred) can now reduce NADP to form NADPH. The electrons are now removed from the system, finding their way to NADPH. The electrons need to be replaced on P840, which requires an external electron donor. In this case, H2S serves as the electron donor. 

 

NOTE: POSSIBLE DISCUSSION

It should be noted that for bacterial photophosphorylation pathways, for each electron donated from a reaction center [remember only one electron is actually donated to the reaction center (or chlorophyl molecule)], the resulting output from that electron transport chain is either the formation of NADPH (requires two electrons) or ATP can be made but NOT not both. In other words, the path the electrons take in the ETC can have one or two possible outcomes. This puts limits on the versatility of the bacterial anoxygenic photosynthetic systems. But what would happen if there evolved a process that utilized both systems, that is, a cyclic and noncyclic photosynthetic pathway in which both ATP and NADPH could be formed from from a single input of electrons? A second limitation is that these bacterial systems require compounds such as reduced sulfur to act as electron donors to reduce the oxidized reaction centers, but they are not necessarily widely found compounds. What would happen if a chlorophyllox molecule would have a reduction potential higher (more positive) than that of the molecular O2/H2O reaction? Answer: a planetary game changer.

 

 

Back to top
S2018_Lecture15_Reading  S2018_Lecture17_Reading
Recommended articles
S2018_Lecture01_Reading
S2018_Lecture02_Reading
S2018_Lecture03_Reading
S2018_Lecture04_Reading
S2018_Lecture05_Reading
The LibreTexts libraries are Powered by MindTouch® and are based upon work supported by the National Science Foundation under grant numbers: 1246120, 1525057, and 1413739. The California State University Affordable Learning Solutions and Merlot are the projects primary partners. Unless otherwise noted, the contents of the LibreTexts library is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 United States License. Permissions beyond the scope of this license may be available at delmarlarsen@gmail.com.
NSF Logo.png   imageedit_7_3300958659.png   imageedit_4_4211606159.png

Login to NB
'''

merged= pd.read_csv("C:\\Users\\ktwic\\Desktop\\with_predictions.csv")
print(merged.columns.values)
generate_heatmap_par(merged)
generate_paragraph_entropy(merged)
# merged["WC"] = merged.text.apply(lambda row: len(word_tokenize(str(row))))
# merged["is_comment"] = merged.parent_id.apply(lambda row: 0 if (row == -1 or pd.isnull(row)) else 1)
# merged["is_question"] = merged.text.apply(lambda row: is_question(row))
# merged["is_comparative"] = merged.text.apply(lambda row: is_comparative(row))
# merged["is_elaboration"] = merged.text.apply(lambda row: is_elaboration(row))

# #merged["parent_label"] = merged.parent_id.apply(lambda row: parent_label(row,merged))
# merged["replies_count"] = merged.comment_id.apply(lambda row : replies_count(row, merged))
# merged["num_sents"] = merged.body.apply(lambda row : len(sent_tokenize(str(row))))
merged["paragraph_ce"] = merged.marked_par.apply(lambda row: calculate_paragraph_ce(merged, row))
# merged["position"] = merged.apply(lambda row: position_in_paragraph(row),axis =1)
# merged["doc_position"] = merged.marked_par.apply(lambda row: position_in_document(str(row)))
# #merged["overlap_ce"] = merged.location_id.apply(lambda row: calculate_overlap_ce(merged, row))
# merged["hashtag_idea"] = merged.body.apply(lambda row : get_hashtag("idea", row))
# merged["hashtag_question"] = merged.body.apply(lambda row : get_hashtag("question", row))
# merged["hashtag_help"] = merged.body.apply(lambda row : get_hashtag("help", row))
# merged["hashtag_useful"] = merged.body.apply(lambda row : get_hashtag("useful", row))
# merged["hashtag_confused"] = merged.body.apply(lambda row : get_hashtag("confused", row))
# merged["hashtag_curious"] = merged.body.apply(lambda row : get_hashtag("curious", row))
# merged["hashtag_interested"] = merged.body.apply(lambda row : get_hashtag("interested", row))
# merged["hashtag_frustrated"] = merged.body.apply(lambda row : get_hashtag("frustrated", row))
#merged["true"] = merged.location_id.apply(lambda row: heatmap_difference(merged, row, True))

# merged["pred"] = merged.location_id.apply(lambda row: heatmap_difference(merged, row, False))
# merged["heatmap_diff"] = abs(merged.true - merged.pred)

merged.to_csv("with_predictions.csv")
#print(anderson_ksamp([merged["true"], merged["pred"]]))
#print(np.mean(merged["heatmap_diff"]), np.std(merged["heatmap_diff"]))