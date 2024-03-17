import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from io import StringIO

def remove_fields(bib_database, fields):
    for entry in bib_database.entries:
        for field in fields:
            if field in entry:
                del entry[field]

def remove_url_when_doi_present(bib_database):
    for entry in bib_database.entries:
        if 'url' in entry and 'doi' in entry:
            del entry['url']

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as bibfile:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibfile, parser=parser)

    hardcoded_entry_strings = [
        """
        @misc{DHPC2022,
            author = {{D}elft {H}igh {P}erformance {C}omputing {C}entre ({DHPC})},
            title = {{D}elft{B}lue {S}upercomputer ({P}hase 1)},
            year = {2022},
            howpublished = {\\url{https://www.tudelft.nl/dhpc/ark:/44463/DelftBluePhase1}},
            ark = {ark:/44463/DelftBluePhase1}
        }
        """,
        """
        @misc{hpc_polito,
            author = {Politecnico di Torino},
            title = {High Performance Computing},
            howpublished = {\\url{https://www.hpc.polito.it/}},
            year = {2011},
        }
        """
    ]

    for entry_string in hardcoded_entry_strings:
        # Parse the hardcoded BibTeX entry string
        string_io = StringIO(entry_string)
        hardcoded_entry_database = bibtexparser.load(string_io)

        # Add the parsed entry to the main BibTeX database
        bib_database.entries.extend(hardcoded_entry_database.entries)

    fields_to_remove = ['urldate', 'issn', 'abstract', 'isbn', 'file', 'note', 'url', 'doi', 'ark']
    remove_fields(bib_database, fields_to_remove)
    # remove_url_when_doi_present(bib_database)

    with open(output_file, 'w', encoding='utf-8') as bibfile:
        bibfile.write(bibtexparser.dumps(bib_database))

if __name__ == "__main__":
    input_file = '/Users/riccardo/Desktop/Cite.bib'
    output_file = 'Cite.bib'
    main(input_file, output_file)
