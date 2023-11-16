import wandb
from neat.math_util import mean, stdev


class WANDB_Reporter(object):

    def start_generation(self, generation):
        self.generation = generation

    def end_generation(self, config, population, species_set):
        self.numberOfSpecies = len(species_set.species)

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)

        wandb.log({
            "epoch": self.generation,
            "numberOfSpecies": self.numberOfSpecies,
            "fit_mean": fit_mean,
            "fit_std": fit_std,
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
