names=['AF-Q3LI64-F1-model_v3.pdb',
 'AF-Q5VTL8-F1-model_v3.pdb',
 'AF-Q15287-F1-model_v3.pdb',
 'AF-Q86TL2-F1-model_v3.pdb',
 'AF-O43316-F1-model_v3.pdb',
 'AF-Q8NBN3-F1-model_v3.pdb',
 'AF-Q8IVB5-F1-model_v3.pdb',
 'AF-L0R819-F1-model_v3.pdb',
 'AF-Q9Y216-F1-model_v3.pdb',
 'AF-Q8NH92-F1-model_v3.pdb',
 'AF-Q9Y259-F1-model_v3.pdb',
 'AF-A6NM03-F1-model_v3.pdb',
 'AF-Q8WXK1-F1-model_v3.pdb',
 'AF-P62070-F1-model_v3.pdb',
 'AF-P12104-F1-model_v3.pdb',
 'AF-P21145-F1-model_v3.pdb',
 'AF-Q9NW38-F1-model_v3.pdb',
 'AF-O60547-F1-model_v3.pdb',
 'AF-Q9NR45-F1-model_v3.pdb',
 'AF-P13716-F1-model_v3.pdb']

num_each_mol = 5
command = "python af2_rg_mc.py"

commands = []
counter = 1
for name in names:
   for i in range(num_each_mol):
      commands.append(f"{command} {name} {i} > af2_rg_mc_structures/logs/{name}_{i}.txt")

with open("Makefile", "w") as f:
  f.write("all:")
  for i in range(len(commands)):
    f.write(f"\tjob{i+1}\t\\\n")
  f.write("\n")
  for i in range(len(commands)):
    f.write(f"job{i+1}:\n\t{commands[i]}\n")
