import wandb
from neat.math_util import mean, stdev
import neat.reporting as reporting
import time


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

class StdOutReporterPeriodique(reporting.BaseReporter):
    """Uses `print` to output information about the run; an example reporter class.
    N'affiche toutes les informations qu'une fois par période"""

    def __init__(self, show_species_detail, periode):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.periode = periode
        self.total_stagnant = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        if self.generation % self.periode == 0:
            ng = len(population)
            ns = len(species_set.species)
            if self.show_species_detail:
                print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
                print("   ID   age  size   fitness   adj fit  stag")
                print("  ====  ===  ====  =========  =======  ====")
                for sid in sorted(species_set.species):
                    s = species_set.species[sid]
                    a = self.generation - s.created
                    n = len(s.members)
                    f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                    af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                    st = self.generation - s.last_improved
                    print(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
            else:
                print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Deleted stagnant : {0:d}'.format(self.total_stagnant))
        self.total_stagnant = 0
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        if self.generation % self.periode != 0:
            return
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        self.total_stagnant += 1

    def info(self, msg):
        pass