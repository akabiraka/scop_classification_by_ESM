import sys
sys.path.append("../scop_classification_by_ESM")

from Bio.PDB import *
from Bio.PDB.PDBIO import Select

class CAAtomSelector(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def  accept_chain(self, chain):
        # print(chain.id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
        
    def accept_atom(self, atom):
        """Overload this to reject atoms for output."""
        if atom.name == "CA":
            return 1
        else:
            return 0        
        
class StandardAminoAcidSelector(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
        
    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if self.chain_id == chain.id:
            return 1
        else:
            return 0
            
    def accept_residue(self, residue):
        if residue.get_resname() in standard_aa_names:
            return 1
        else:
            return 0

class AllBackboneAtomSelector(Select):
    """Backbone atoms: CA, CB, N, O

    Args:
        Select ([type]): [description]
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def  accept_chain(self, chain):
        # print(chain.id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
        
    def accept_atom(self, atom):
        """Overload this to reject atoms for output."""
        if atom.name == "CA":
            return 1
        elif atom.name == "CB":
            return 1
        elif atom.name == "N":
            return 1
        elif atom.name == "O":
            return 1
        else:
            return 0
        
class ChainAndAminoAcidSelect(Select):
    def __init__(self, chain_id):
        super(ChainAndAminoAcidSelect).__init__()
        self.chain_id = chain_id
        
    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if chain.id == self.chain_id:
            return 1
        else:
            return 0
            
    def accept_residue(self, residue):
        hetero_flag, sequence_identifier, insertion_code = residue.id
        if residue.get_resname() in standard_aa_names:# and insertion_code==" ":
            # print(residue.get_resname())
            return 1
        else:
            return 0

class ChainAndRegionSelect(Select):
    def __init__(self, chain_id, region) -> None:
        """
        Args:
            chain_id (char): "A"
            region (str): range in string inclusive, format example "2-113"
        """
        super(ChainAndRegionSelect, self).__init__()
        self.chain_id = chain_id
        if region.count("-")==1: 
            start, end = region.split("-")
        elif region.count("-")==2: 
            loc = region.index("-", region.index("-")+1)
            start, end = region[:loc], region[loc+1:]
        elif region.count("-")==3:
            loc = region.index("-", region.index("-")+1)
            start, end = region[:loc], region[loc+1:]
            # raise NotImplementedError()

        # print(start, end)
        if start.startswith("-"):
            self.start_residue_id = (" ", -int(start[1:]), " ")    
        elif start.isdigit():
            self.start_residue_id = (" ", int(start), " ")
        else: 
            self.start_residue_id = (" ", int(start[:-1]), start[-1])

        if end.startswith("-"):
            self.end_residue_id = (" ", -int(end[1:]), " ")        
        elif end.isdigit():
            self.end_residue_id = (" ", int(end), " ")
        else: 
            self.end_residue_id = (" ", int(end[:-1]), end[-1])
        
        # print(self.start_residue_id)
        # print(self.end_residue_id)
        

    def  accept_chain(self, chain):
        # print(chain.id, self.chain_id)
        if chain.id == self.chain_id:
            # creating region residue list
            flag = False
            self.region_residue_list = []
            for residue in chain.get_residues():
                # print(residue.id)
                hetero_flag, seq_id, insertion_code = residue.id
                if seq_id==self.start_residue_id[1] and insertion_code==self.start_residue_id[2]: 
                    flag=True
                elif seq_id==self.end_residue_id[1] and insertion_code==self.end_residue_id[2]:
                    self.region_residue_list.append(residue)
                    break
                if flag: self.region_residue_list.append(residue)
            # print(self.region_residue_list)
            return 1
        else:
            return 0
    
    def accept_residue(self, residue):
        if residue.get_resname() in standard_aa_names and residue in self.region_residue_list:
            # print(residue.get_resname())
            return 1
        else:
            return 0
