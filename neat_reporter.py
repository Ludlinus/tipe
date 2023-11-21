import wandb
from neat.math_util import mean, stdev


class WANDB_Reporter(object):

    def start_generation(self, generation):
        self.generation = generation

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]
        #fit_mean = mean(fitnesses)
        #fit_std = stdev(fitnesses)
        #nbOfSpecies = len(species_set.species)

        wandb.log({
            "epoch": self.generation,
            "fit_mean": mean(fitnesses),
            "fit_std": stdev(fitnesses),
            "nbOfSpecies":len(species.species)
        })

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass
