import io
import pickle


class RenameUnpickler(pickle.Unpickler):
    def set_rename_data(self,old_mod,new_mod):
        self.old_mod = old_mod
        self.new_mod = new_mod

    def find_class(self, module, name):
        renamed_module = module
        if module == self.old_mod:
            try:
                return super(RenameUnpickler, self).find_class(renamed_module, name)
            except:
                renamed_module = self.new_mod
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj,old_mod,new_mod):
    ru = RenameUnpickler(file_obj)
    ru.set_rename_data(old_mod,new_mod)
    return ru.load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)