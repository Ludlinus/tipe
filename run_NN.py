import neat
import tqdm

import neat_reporter
import wandb

wandb_API = wandb.Api()
# sweep = wandb_API.project("sweat_pas_rose/TIPE-2").sweeps()[0]
sweep = wandb_API.sweep("sweat_pas_rose/TIPE-2/8vbeeuy7")
sweep_id = sweep.id


class TrainingCycle:
    def __init__(self, taille, pos1, pos2, label1, label2):  # label 1/2: identifiant de l'agent 1/2
        self.taille = taille
        self.pos1, self.pos2 = pos1, pos2
        self.label1, self.label2 = label1, label2


liste_graphes = [
    TrainingCycle(84, 39, 77, 23, 62),
    TrainingCycle(11, 3, 7, 39, 56),
    TrainingCycle(73, 18, 38, 85, 20),
    TrainingCycle(53, 22, 48, 6, 7),
    TrainingCycle(65, 28, 59, 52, 98),
    TrainingCycle(27, 7, 20, 23, 27),
    TrainingCycle(59, 3, 47, 78, 18),
    TrainingCycle(12, 1, 9, 78, 79),
    TrainingCycle(12, 3, 6, 77, 3),
]
liste_graphes_supplementaires = [
    TrainingCycle(96, 10, 91, 91, 0),
    TrainingCycle(15, 2, 13, 30, 92),
    TrainingCycle(65, 27, 37, 43, 39),
    TrainingCycle(13, 0, 8, 64, 36),
    TrainingCycle(24, 11, 14, 54, 44),
    TrainingCycle(61, 18, 44, 50, 86),
    TrainingCycle(74, 14, 56, 36, 76),
    TrainingCycle(84, 37, 58, 73, 92),
    TrainingCycle(20, 0, 19, 17, 80),
    TrainingCycle(99, 25, 58, 26, 58),
    TrainingCycle(52, 12, 32, 69, 95),
    TrainingCycle(62, 0, 57, 0, 31),
    TrainingCycle(22, 2, 18, 98, 60),
    TrainingCycle(28, 5, 24, 7, 9),
    TrainingCycle(13, 2, 11, 16, 18),
    TrainingCycle(31, 2, 24, 95, 93),
    TrainingCycle(15, 5, 10, 76, 51),
    TrainingCycle(63, 13, 60, 5, 51),
    TrainingCycle(99, 43, 59, 92, 1),
    TrainingCycle(46, 6, 30, 78, 41),
    TrainingCycle(49, 23, 48, 26, 79),
    TrainingCycle(82, 4, 59, 40, 0),
    TrainingCycle(31, 9, 28, 70, 5),
    TrainingCycle(21, 4, 14, 64, 78),
    TrainingCycle(77, 34, 70, 72, 83),
    TrainingCycle(53, 16, 34, 5, 95),
    TrainingCycle(17, 8, 13, 35, 68),
    TrainingCycle(55, 23, 51, 71, 76),
    TrainingCycle(17, 2, 15, 35, 90),
    TrainingCycle(88, 22, 65, 47, 19),
    TrainingCycle(81, 10, 46, 29, 38),
    TrainingCycle(29, 1, 23, 38, 97),
    TrainingCycle(37, 2, 22, 30, 87),
    TrainingCycle(65, 1, 65, 18, 96),
    TrainingCycle(32, 11, 30, 17, 16),
    TrainingCycle(97, 16, 55, 67, 6),
    TrainingCycle(10, 3, 5, 37, 85),
    TrainingCycle(67, 4, 50, 57, 88),
    TrainingCycle(39, 2, 24, 85, 26),
    TrainingCycle(42, 8, 27, 64, 26),
    TrainingCycle(22, 3, 13, 8, 69)
]


def eval_genomes(genomes, config):
    for genomes_id, genome in tqdm.tqdm(genomes):
        genome.fitness = 0
        for graphe in liste_graphes:
            label1, label2 = graphe.label1, graphe.label2  # label: identifiant dans le graphe évalué
            taille_cycle = graphe.taille  # taille du graphe évalué
            iterations_max = 1_000

            # agent1 = ag.Agent_NN(genome=genome, label=label1, config=config)
            # agent2 = ag.Agent_NN(genome=genome, label=label2, config=config)

            etape = 0
            derniereAction_1 = 0
            derniereAction_2 = 0

            nn1 = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
            nn2 = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)

            pos_1 = graphe.pos1
            pos_2 = graphe.pos2

            # Evaluation des réseaux
            iterations = 0
            dist_min = abs(pos_1 - pos_2)  # distance de départ entre 1 et 2
            while pos_1 != pos_2 and iterations <= iterations_max:  # stoppe au bout de 1000 itérations

                out1 = nn1.activate([label1, etape, pos_1, derniereAction_1])
                out2 = nn2.activate([label2, etape, pos_2, derniereAction_2])

                derniereAction_1 = max(range(len(out1)), key=out1.__getitem__) - 1
                derniereAction_2 = max(range(len(out2)), key=out2.__getitem__) - 1

                # derniereAction_1 = max(range(len(out1)), key=out1.__getitem__)
                # derniereAction_2 = max(range(len(out2)), key=out2.__getitem__)

                pos_1 += derniereAction_1  # les agents se déplacent ou pas
                pos_2 += derniereAction_2

                pos_1 %= taille_cycle  # Si les agents ont fait le tour du graphe, leur position revient à 0
                pos_2 %= taille_cycle

                if abs(pos_1 - pos_2) < dist_min:
                    dist_min = abs(pos_1 - pos_2)  # on enregistre la distance min atteinte par les agents

                iterations += 1

            # Mise à jour de la fitness

            if iterations <= iterations_max:
                genome.fitness -= iterations
            else:
                genome.fitness -= iterations_max * dist_min


def entrainement():
    run_name = sorted([run.name for run in sweep.runs], reverse=True)[0]
    run_name = run_name[:run_name.rfind('.')] + "." + str(int(run_name[run_name.rfind('.') + 1:]) + 1)

    wandb.init(name=run_name)

    config_file = "config_1.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Chargement des hyper-parametres spécifiques depuis WANDB

    config.genome_config.node_add_prob = wandb.config.node_add_prob
    config.genome_config.node_delete_prob = wandb.config.node_delete_prob

    config.genome_config.conn_add_prob = wandb.config.conn_add_prob
    config.genome_config.conn_delete_prob = wandb.config.conn_delete_prob

    config.genome_config.bias_max_value = wandb.config.bias_max_value
    config.genome_config.bias_min_value = -wandb.config.bias_max_value

    config.genome_config.weight_max_value = wandb.config.weight_max_value
    config.genome_config.weight_min_value = -wandb.config.weight_max_value

    config.reproduction_config.survival_threshold = wandb.config.survival_threshold

    p = neat.Population(config)

    p.add_reporter(neat_reporter.WANDB_Reporter())
    p.add_reporter(neat.checkpoint.Checkpointer(generation_interval=500, time_interval_seconds=None,
                                                filename_prefix='saves/neat-checkpoint-'))

    artifactToSave = wandb.Artifact(name="neat_checkpoints_" + str(wandb.run.name), type="neat_checkpoints")
    artifactToSave.add_dir("saves")
    artifactToSave.save()

    run_result = p.run(eval_genomes, 1_001)  # 1000 +1 pour être certain d'enregistrer le dernier checkpoint

    wandb.finish()


def main():
    wandb.agent(sweep_id=sweep_id, function=entrainement, project="TIPE-2", entity="sweat_pas_rose")


if __name__ == '__main__':
    main()
