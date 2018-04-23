import java.util.*;

class Puzzle{
    public static LinkedList<Stare> stariProcesate;
    public static LinkedList<Stare> stariPentruProcesare;
    public static int pozitionareInitiala[][];
    public static int pozitionareFinala[][];
    public static HashMap<Integer, IJPos> pozCifreStareFinala;
    public static final int N = 3;

    public static void main(String args[]){
        stariProcesate = new LinkedList<Stare>();
        stariPentruProcesare = new LinkedList<Stare>();

        pozitionareInitiala = new int[][]{
                {8, 1, 3},
                {4,-1, 2},
                {7, 6, 5}
        };

        pozitionareFinala = new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8,-1}
        };

        generPozCifreStareFinala(pozitionareFinala);

        stariPentruProcesare.add(new Stare(null,0,calcEuristica(pozitionareInitiala),pozitionareInitiala));

        while(stariPentruProcesare.size()>0){
            Stare stareProcesataAcum = stariPentruProcesare.poll();

            if(comparaDouaPozitionari(stareProcesataAcum.pozitionare, pozitionareFinala)==true){
                LinkedList<Stare> istoriaMutarilor = new LinkedList<Stare>();

                while(stareProcesataAcum!=null){
                    istoriaMutarilor.add(stareProcesataAcum);
                    stareProcesataAcum = stareProcesataAcum.starePrecedenta;
                }

                int numarDeMutari = istoriaMutarilor.size()-1;
                while(istoriaMutarilor.size()>0){
                    stareProcesataAcum = istoriaMutarilor.removeLast();
                    afisarePozitionare(stareProcesataAcum);
                }
                System.out.println("total mutari: "+numarDeMutari);
                break;
            }

            stariPentruProcesare.addAll(genereazaStariApropiate(stareProcesataAcum));

            Collections.sort(stariPentruProcesare, new Comparator<Stare>(){
               public int compare(Stare stare1, Stare stare2){
                   return (stare1.nrPasiPanLaStare+stare1.euristica) - (stare2.nrPasiPanLaStare+stare2.euristica);
               }
            });

            stariProcesate.add(stareProcesataAcum);
        }

    }

    public static boolean comparaDouaPozitionari(int primaPoz[][], int aDouaPoz[][]){
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                if(primaPoz[i][j]!=aDouaPoz[i][j])
                    return false;
        return true;
    }

    public static void afisarePozitionare(Stare stare){
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                if(stare.pozitionare[i][j]==-1)
                    System.out.print("_ ");
                else
                    System.out.print(stare.pozitionare[i][j]+" ");
            }
            System.out.print("\n");
        }
        System.out.println("---");
    }

    public static LinkedList<Stare> genereazaStariApropiate(Stare stareDinCareGeneram){
        IJPos pozSpatiu = null;
        LinkedList<Stare> stariApropiate = new LinkedList<Stare>();

        for(int i=0;i<N;i++){//gasim pozitia lui -1 care inseamna spatiu
            for(int j=0;j<N;j++){
                if(stareDinCareGeneram.pozitionare[i][j]==-1) {
                    pozSpatiu = new IJPos(i, j);
                    break;
                }
            }
            if(pozSpatiu!=null) break;
        }

        if(pozSpatiu.j-1 >= 0)
            copieSiAdaugaStarea(stareDinCareGeneram, stariApropiate, pozSpatiu, 0,-1);

        if(pozSpatiu.j+1<N)
            copieSiAdaugaStarea(stareDinCareGeneram, stariApropiate, pozSpatiu, 0,1);

        if(pozSpatiu.i-1 >= 0)
            copieSiAdaugaStarea(stareDinCareGeneram, stariApropiate, pozSpatiu, -1,0);

        if(pozSpatiu.i+1 < N)
            copieSiAdaugaStarea(stareDinCareGeneram, stariApropiate, pozSpatiu, 1,0);

        return stariApropiate;
    }

    public static void copieSiAdaugaStarea(Stare stareDinCareGeneram, LinkedList<Stare> stariApropiate,
                                           IJPos pozSpatiu, int mutareI, int mutareJ){
        int oPozitionareNoua[][] = new int[N][N];

        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                oPozitionareNoua[i][j] = stareDinCareGeneram.pozitionare[i][j];

        swap(oPozitionareNoua, pozSpatiu, mutareI, mutareJ);

        Iterator itStariProcesate = stariProcesate.iterator();

        boolean pozitionareNoua = true;
        while(itStariProcesate.hasNext()){
            int pozitionarePentruComparare[][] = ((Stare)itStariProcesate.next()).pozitionare;
            if(comparaDouaPozitionari(oPozitionareNoua, pozitionarePentruComparare)==true){
                pozitionareNoua = false;
                break;
            }
        }

        if(pozitionareNoua)
            stariApropiate.add(new Stare(stareDinCareGeneram, stareDinCareGeneram.nrPasiPanLaStare+1,
                calcEuristica(oPozitionareNoua), oPozitionareNoua));
    }

    public static void swap(int stare[][], IJPos pozSpatiu, int mutareI, int mutareJ){
        int iNou = pozSpatiu.i + mutareI;
        int jNou = pozSpatiu.j + mutareJ;
        int tmp = stare[pozSpatiu.i][pozSpatiu.j];
        stare[pozSpatiu.i][pozSpatiu.j] = stare[iNou][jNou];
        stare[iNou][jNou] = tmp;
    }

    public static int calcEuristica(int pozitionare[][]){
        int euristica = 0;

        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                int cifra = pozitionare[i][j];
                if(cifra>-1){
                    IJPos pozsCifra = pozCifreStareFinala.get(cifra);
                    int eurTmp = Math.abs(pozsCifra.i - i) + Math.abs(pozsCifra.j - j);
                    euristica += eurTmp;
                }
            }
        }

        return euristica;
    }

    public static void generPozCifreStareFinala(int stare[][]){
        pozCifreStareFinala = new HashMap<Integer, IJPos>();
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                pozCifreStareFinala.put((Integer)(stare[i][j]), new IJPos(i, j));
            }
        }
    }
}

class Stare{
    Stare starePrecedenta;
    int nrPasiPanLaStare;
    int euristica;

    int pozitionare[][];

    public Stare(Stare stPrec, int nrPsPnLaSt, int eur, int poz[][]){
        this.starePrecedenta = stPrec;
        this.nrPasiPanLaStare = nrPsPnLaSt;
        this.euristica = eur;
        this.pozitionare = poz;
    }
}

class IJPos{
    int i,j;
    public IJPos(int i, int j){
        this.i = i;
        this.j = j;
    }
}