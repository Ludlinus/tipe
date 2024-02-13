import os

import neat
import tqdm

import neat_reporter
import wandb

wandb_API = wandb.Api()
sweep = wandb_API.sweep("sweat_pas_rose/TIPE-2/12mjhjtj")
sweep_id = sweep.id

counter = 20

try:
    os.mkdir("./saves")
except FileExistsError:
    pass


class TrainingCycle:
    def __init__(self, taille, pos1, pos2, label1, label2):  # label 1/2: identifiant de l'agent 1/2
        self.taille = taille
        self.pos1, self.pos2 = pos1, pos2
        self.label1, self.label2 = label1, label2

    def starting_distance(self):
        return min(abs(self.pos1 - self.pos2), abs(self.pos1 - self.taille) + abs(self.pos2 - self.taille))


liste_graphes = [
    TrainingCycle(10, 3, 7, 39, 56),
    TrainingCycle(20, 18, 5, 85, 20),
    TrainingCycle(30, 22, 7, 6, 7),
]


def eval_genomes(genomes, config):
    iterations_max = 1_000

    for genomes_id, genome in tqdm.tqdm(genomes):
        genome.fitness = 0
        for graphe in liste_graphes:
            label1, label2 = graphe.label1, graphe.label2  # label: identifiant dans le graphe évalué
            taille_cycle = graphe.taille  # taille du graphe évalué

            etape = 0
            derniereAction_1 = 0
            derniereAction_2 = 0

            nn1 = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
            nn2 = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)

            pos_1 = graphe.pos1
            pos_2 = graphe.pos2

            # Evaluation des réseaux
            iterations = 0
            dist_min = graphe.starting_distance()  # distance de départ entre 1 et 2
            while pos_1 != pos_2 and iterations <= iterations_max:  # stoppe au bout de 1000 itérations

                out1 = nn1.activate([label1, etape, pos_1, derniereAction_1])
                out2 = nn2.activate([label2, etape, pos_2, derniereAction_2])

                derniereAction_1 = max(range(len(out1)), key=out1.__getitem__) - 1
                derniereAction_2 = max(range(len(out2)), key=out2.__getitem__) - 1


                pos_1 += derniereAction_1  # les agents se déplacent ou pas
                pos_2 += derniereAction_2

                pos_1 %= taille_cycle  # Si les agents ont fait le tour du graphe, leur position revient à 0
                pos_2 %= taille_cycle

                if abs(pos_1 - pos_2) < dist_min:
                    dist_min = min(abs(pos_1 - pos_2), abs(pos_1 - taille_cycle) + abs(
                        pos_2 - taille_cycle))  # on enregistre la distance min atteinte par les agents

                iterations += 1

            # Mise à jour de la fitness

            if iterations <= iterations_max:
                genome.fitness -= iterations
            else:
                genome.fitness -= iterations_max * dist_min

def entrainement():
    run_name = sorted([run.name for run in sweep.runs], reverse=True)[1]
    run_name = run_name[:run_name.rfind('.')] + "." + str(int(run_name[run_name.rfind('.') + 1:]) + 1)

    wandb.init(name=run_name)

    config_file = "config_1.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Chargement des hyper-parametres spécifiques depuis WANDB

    config.genome_config.node_add_prob = wandb.config.node_add_prob
    config.genome_config.node_delete_prob = wandb.config.node_delete_prob

    config.reproduction_config.survival_threshold = wandb.config.survival_threshold
    config.pop_size = wandb.config.pop_size

    p = neat.Population(config)

    p.add_reporter(neat_reporter.WANDB_Reporter())
    p.add_reporter(neat.checkpoint.Checkpointer(generation_interval=500, time_interval_seconds=None,
                                                filename_prefix='saves/neat-checkpoint-'))
    artifactToSave = wandb.Artifact(name="neat_checkpoints_" + str(wandb.run.name), type="neat_checkpoints")
    artifactToSave.add_dir("saves")
    artifactToSave.save()

    run_result = p.run(eval_genomes, 10_000 + 1)  # 1000 +1 pour être certain d'enregistrer le dernier checkpoint

    wandb.finish()

def main():
    wandb.agent(sweep_id=sweep_id, function=entrainement, project="TIPE-2", entity="sweat_pas_rose", count=counter)


if __name__ == '__main__':
    main()
