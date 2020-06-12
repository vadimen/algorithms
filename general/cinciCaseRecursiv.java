//problema cu cinci case http://www.gandul.info/magazin/testul-de-inteligenta-al-lui-albert-einstein-10411503

import java.util.*;

class CinciCase{
    public static ArrayList<Permutare> permutariCulori;
    public static ArrayList<Permutare> permutariNationalitati;
    public static ArrayList<Permutare> permutariBauturi;
    public static ArrayList<Permutare> permutariTigari;
    public static ArrayList<Permutare> permutariAnimale;
    public static final int N = 5;

    public static void main(String args[]){
        long timpInceput = System.currentTimeMillis();

        permutariCulori        = new ArrayList<Permutare>();
        permutariNationalitati = new ArrayList<Permutare>();
        permutariBauturi       = new ArrayList<Permutare>();
        permutariTigari        = new ArrayList<Permutare>();
        permutariAnimale       = new ArrayList<Permutare>();

        Permutare combRez[] = new Permutare[N];

        String culori[]           = new String[]{"alba", "verde", "rosie", "albastra", "galbena"};
        String nationalitati[]    = new String[]{"suedez","german","britanic","danez","norvegian"};
        String bauturi[]          = new String[]{"bere", "cafea", "lapte", "ceai", "apa"};
        String tigari[]           = new String[]{"winfield", "rothmans", "pall mall","marlboro", "dunhill"};
        String animale[]          = new String[]{"caine", "peste", "pasare", "cal", "pisica"};

//        Collections.reverse(Arrays.asList(culori));
//        Collections.reverse(Arrays.asList(nationalitati));
//        Collections.reverse(Arrays.asList(bauturi));
//        Collections.reverse(Arrays.asList(tigari));
//        Collections.reverse(Arrays.asList(animale));

        genereazaPermutari(0, new String[N], culori, permutariCulori);
        genereazaPermutari(0, new String[N], nationalitati, permutariNationalitati);
        genereazaPermutari(0, new String[N], bauturi, permutariBauturi);
        genereazaPermutari(0, new String[N], tigari, permutariTigari);
        genereazaPermutari(0, new String[N], animale, permutariAnimale);

        long contor = 0;
        for(int i=0;i<permutariCulori.size();i++){
            combRez[0] = permutariCulori.get(i);
            for(int j=0;j<permutariNationalitati.size();j++){
                combRez[1] = permutariNationalitati.get(j);
                for(int k=0;k<permutariBauturi.size();k++){
                    combRez[2] = permutariBauturi.get(k);
                    for(int g=0;g<permutariTigari.size();g++){
                        combRez[3] = permutariTigari.get(g);
                        for(int h=0;h<permutariAnimale.size();h++){
                            combRez[4] = permutariAnimale.get(h);
                            afisareCombinatie(combRez);
                            System.out.println(++contor);
                            if(combRezValida(combRez)){
                                System.out.println("Rezultat gasit");
                                System.out.println("Executat in " +
                                        (System.currentTimeMillis()-timpInceput)/1000+" secunde.");
                                return;
                            }
                        }
                    }
                }
            }
        }

        System.out.println("Executat in "+(System.currentTimeMillis()-timpInceput)/1000+" secunde.");
    }

    public static void afisareCombinatie(Permutare combinatie[]){
        int i;
        for(i=0;i<N;i++) {System.out.print(combinatie[0].permutare[i]+" ");}
        System.out.println();
        for(i=0;i<N;i++) {System.out.print(combinatie[1].permutare[i]+" ");}
        System.out.println();
        for(i=0;i<N;i++) {System.out.print(combinatie[2].permutare[i]+" ");}
        System.out.println();
        for(i=0;i<N;i++) {System.out.print(combinatie[3].permutare[i]+" ");}
        System.out.println();
        for(i=0;i<N;i++) {System.out.print(combinatie[4].permutare[i]+" ");}
        System.out.println();
        System.out.println("---------------------");
    }

    public static boolean combRezValida(Permutare combinatiePermutari[]){
        if(!combinatiePermutari[2].permutare[2].equals("lapte"))
            return false;

        if(!combinatiePermutari[1].permutare[0].equals("norvegian"))
            return false;

        if(!combinatiePermutari[0].permutare[1].equals("albastra"))
            return false;

        int i;
        for(i=0;i<N;i++)
            if(combinatiePermutari[3].permutare[i].equals("marlboro"))
                break;

        boolean marlboroOk = false;
        if(i-1>=0)
            if(combinatiePermutari[2].permutare[i-1].equals("apa"))
                marlboroOk = true;

        if(!marlboroOk){
            if(i+1<N)
                if(combinatiePermutari[2].permutare[i+1].equals("apa"))
                    marlboroOk = true;
        }
        if(!marlboroOk)
            return false;

        marlboroOk = false;
        if(i-1>=0)
            if(combinatiePermutari[4].permutare[i-1].equals("pisica"))
                marlboroOk = true;

        if(!marlboroOk){
            if(i+1<N)
                if(combinatiePermutari[4].permutare[i+1].equals("pisica"))
                    marlboroOk = true;
        }
        if(!marlboroOk)
            return false;

        return true;
    }

    public static void genereazaPermutari(int poz, String rezultat[], String date[], ArrayList<Permutare> colectie){
        if(poz == N){
            colectie.add(new Permutare(rezultat));
            for(int j=0;j<N;j++)
                System.out.print(rezultat[j]+" ");
            System.out.println();
        }else {
            for (int i = 0; i < N; i++) {
                rezultat[poz] = date[i];
                if(permutareValida(poz, rezultat)){
                    genereazaPermutari(poz+1, rezultat, date, colectie);
                }
            }
        }
    }

    public static boolean permutareValida(int poz, String rezultat[]){
        for(int i=0;i<poz;i++)
            if(rezultat[i].compareTo(rezultat[poz])==0)
                return false;
        return true;
    }

    static class Permutare{
        String permutare[];
        public Permutare(String vector[]){
            permutare = new String[N];
            for(int i=0;i<N;i++) permutare[i] = vector[i];
        }
    }
}
