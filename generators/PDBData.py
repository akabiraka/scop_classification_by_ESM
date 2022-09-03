import sys
sys.path.append("../scop_classification_by_ESM")


import requests
from Bio.PDB import *
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import PPBuilder 
from Bio import SeqIO

class PDBData(object):
    def __init__(self, pdb_dir="data/pdbs/"):
        super(PDBData, self).__init__()
        self.pdb_dir = pdb_dir
        self.pdb_format = "mmCif"
        self.pdb_ext = ".cif"
        self.parser = MMCIFParser(QUIET=True)
        self.pdbl = PDBList()
        self.pdbio = PDBIO()
        self.ppb=PPBuilder()
        
    def validate_pdb_id(self, pdb_id):
        """
        Validate the given pdb_id which must be a 4 letter pdb_id. 
        Else raise an Exception.
        """
        if len(pdb_id) == 4:
            return pdb_id
        elif len(pdb_id) > 4:
            pdb_id = pdb_id[0:4].lower()
        else:
            raise Exception("Reported ", pdb_id, "is not a pdb id in 4 letter format.")
        return pdb_id
        
    def download_structure(self, pdb_id):
        """Download the pdb structure into mmCif format.

        Args:
            pdb_id (string): pdb_id
        """
        self.pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_dir, file_format=self.pdb_format)
    
    def get_first_chain_id(self, pdb_id):
        pdb_file = self.pdb_dir + pdb_id + self.pdb_ext
        structure = self.parser.get_structure("", pdb_file)[0]
        print(list(structure.get_chains()))
        return list(structure.get_chains())[0].id

    def get_A_or_first_chain_id(self, pdb_id):
        cln_pdb_file = self.pdb_dir + pdb_id + self.pdb_ext
        structure = self.parser.get_structure("", cln_pdb_file)[0]
        chain_ids = list(structure.get_chains())
        print(chain_ids)
        for chain_id in chain_ids:
            if chain_id.id == "A": return "A"
        return chain_ids[0].id
    
    def clean(self, inp_pdb_filepath, out_pdb_filepath=None, selector=None):
        """Given a Select instance, this function reads a mmCif protein data and saves into pdb format.
        Args:
            inp_pdb_filepath (str): i.e "pdbs/1lve.cif"
            out_pdb_filepath (str): i.e "pdbs_clean/1lveA.pdb"
            selector: Subclass of Bio.PDB.Select class
        Returns:
            PDB Structure
        """
        print(f"Cleaning {inp_pdb_filepath}")
        structure = self.parser.get_structure("", inp_pdb_filepath)[0]
        self.pdbio.set_structure(structure)
        
        if selector is None: self.pdbio.save(out_pdb_filepath)
        else: self.pdbio.save(out_pdb_filepath, select=selector)
        
        return PDBParser(QUIET=True).get_structure("", out_pdb_filepath)
            
    def get_chain_from_structure(self, structure, chain_id):
        """Given a structure and chain_id, it returns 
        corresponding chain of the structure

        Args:
            structure ([Bio.PDB.Structure]): [An instance of Bio.PDB.Structure]
            chain_id ([character]): [chain_id]

        Returns:
            chain: Bio.PDB.Chain
        """
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        for chain in chains:
            if chain.id == chain_id:
                return chain
            
    def get_chain_from_clean_pdb(self, pdb_id, chain_id, pdb_filename):
        """Return chain given a pdb file.

        Args:
            pdb_id (string)
            chain_id (character)
            pdb_filename (path): A pdb file path. i.e: "data/pdbs_clean/1c8cA.pdb"
        Returns:
            Chain: A chain of a Protein data
        """
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_filename)
        models = list(structure.get_models())
        chains = list(models[0].get_chains())
        # for each chain
        for chain in chains:
            if chain.id == chain_id:
                return chain
            
    def generate_fasta_from_pdb(self, pdb_id, chain_id, input_pdb_filepath, out_fasta_file=None):
        """Return sequence and length, and create fasta file from pdb data.

        Args:
            pdb_id (string)
            chain_id (string)
            input_pdb_filepath (string): the path must have a pdb file. i.e data/pdbs_clean/1amqA.pdb
        Returns:
            seq: primary sequence of the pdb file
        """
        
        structure = PDBParser(QUIET=True).get_structure(pdb_id, input_pdb_filepath)
        residues = structure[0][chain_id].get_residues()
        seq = ""
        for residue in residues:
            seq += Polypeptide.three_to_one(residue.get_resname())
        
        if out_fasta_file is not None:
            with open(out_fasta_file, "w") as fasta_file_handle:
                fasta_file_handle.write(">{}:{}\n".format(pdb_id.upper(), chain_id))
                fasta_file_handle.write(seq)
        print("Generating fasta {}:{}, Seq-len:{} ... ..".format(pdb_id, chain_id, len(seq)))
        return seq, len(seq)
    
    def create_mutant_fasta_file(self, wild_fasta_file, mutant_fasta_file, zero_based_mutation_site, mutant_residue, mutation=None):
        pdbid = wild_fasta_file.split("/")[2].split(".")[0]
        mutant_residue = Polypeptide.three_to_one(mutant_residue) if len(mutant_residue)==3 else mutant_residue
        with open(wild_fasta_file, "r") as wild_fasta_reader:
            lines = wild_fasta_reader.readlines()
            # print(lines)
            with open(mutant_fasta_file, 'w') as mutant_fasta_writer:
                # print(lines[1][mutation_site])# = mutant_residue
                fasta = lines[1][:zero_based_mutation_site] + mutant_residue + lines[1][zero_based_mutation_site+1:]
                mutant_fasta_writer.write(lines[0].rstrip()+":{}\n".format(mutation))
                mutant_fasta_writer.write(fasta)
    
    def __save_fasta(self, pdb_id, fasta_text, is_save_file=True, fasta_dir="data/fastas/"):
        """Private method for download_fasta method

        Args:
            pdb_id (string): pdb_id
            fasta_text (string): text in fasta format
            is_save_file (bool, optional): [description]. Defaults to True.
            fasta_dir (str, optional): Directory path. Defaults to "data/fastas/".
        """
        if is_save_file:
            filepath = "{}{}_downloaded.fasta".format(fasta_dir, pdb_id)
            with open(filepath, "w") as fasta_file_handle:
                fasta_file_handle.write(fasta_text)
                fasta_file_handle.close()
        
    def download_fasta(self, pdb_id, is_save_file=True, fasta_dir="data/fastas/"):
        """Download fasta from pdb using pdb_id

        Args:
            pdb_id (string): pdb_id
            is_save_file (bool, optional): Self-explained. Defaults to True.
            fasta_dir (str, optional): Directory path. Defaults to "data/fastas/".
        """
        print("Downloading fasta {} ... ..".format(pdb_id))
        http = requests.Session()
        save_fasta_hook = lambda response, *args, **kwargs: self.__save_fasta(pdb_id, response.text, is_save_file, fasta_dir)
        http.hooks["response"] = save_fasta_hook
        http.get('https://www.rcsb.org/fasta/entry/'+pdb_id)
        
    def get_fasta_seq_record(self, fasta_file):
        """
        Returns:
            [Bio.SeqRecord]: returns the 1st Bio.SeqRecord from fasta file.
        """
        return next(SeqIO.parse(fasta_file, "fasta"))
        
    
    def get_seq_from_pdb(self, input_pdb_filepath):   
        """Given a input pdb filepath, it returns the sequence

        Args:
            input_pdb_filepath (string): example: "data/pdbs_clean/1amqA.pdb"

        Returns:
            Bio.Seq.Seq
        """
        structure = PDBParser(QUIET=True).get_structure("id", input_pdb_filepath)
        polypeptide = self.ppb.build_peptides(structure).__getitem__(0)
        return polypeptide.get_sequence()
    
    def get_first_residue_id(self, pdb_file, chain_id):
        residue_list = list(PDBParser(QUIET=True).get_structure("", pdb_file)[0][chain_id].get_residues())
        _, residue_id, _ = residue_list[0].id
        return residue_id
                
    def get_last_residue_id(self, pdb_file, chain_id):
        residue_list = list(PDBParser(QUIET=True).get_structure("", pdb_file)[0][chain_id].get_residues())
        _, residue_id, _ = residue_list[len(residue_list)-1].id
        return residue_id
    
    def get_residue_ids_dict(self, pdb_file, chain_id):
        residues = PDBParser(QUIET=True).get_structure("", pdb_file)[0][chain_id].get_residues()
        residue_ids_dict = {residue.id:i for i, residue in enumerate(residues)}
        return residue_ids_dict
    
    def get_zero_based_mutation_site(self, cln_pdb_file, chain_id, mutation_residue_id):
        residue_ids_dict = self.get_residue_ids_dict(pdb_file=cln_pdb_file, chain_id=chain_id)
        return residue_ids_dict.get(mutation_residue_id)

    def does_mutation_site_has_expected_residue(self, cln_pdb_file, chain_id, mutation_residue_id, residue_name):
        residue = PDBParser(QUIET=True).get_structure("", cln_pdb_file)[0][chain_id][mutation_residue_id]
        pdb_residue_name = Polypeptide.three_to_one(residue.get_resname())
        return pdb_residue_name == residue_name, pdb_residue_name


# pdb_id="1h7m"    
# chain_id="A"
# cln_pdb_file = "data/pdbs_clean/1h7mA.pdb"
# pdbdata = PDBData(pdb_dir="data/pdbs/")
# residue_ids_dict = pdbdata.get_residue_ids_dict(pdb_file=cln_pdb_file, chain_id=chain_id)
# print(residue_ids_dict)
# pdbdata.get_full_ss_and_sa(pdb_id, chain_id, cln_pdb_file, ss_dict, sa_classification_func=get_sa_type)