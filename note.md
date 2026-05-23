## SETUP E CONFIGURAZIONE

L'esercizio è stato eseguito su singola macchina in modalità local[*], dove Spark simula un cluster distribuito tramite thread worker locali. Questo permette di valutare correttezza algoritmica, scalabilità del problema in n, e l'efficienza di parallelizzazione thread-level, ma non riproduce le caratteristiche di un cluster vero (overhead di rete, fault tolerance, replica dei dati). I tempi misurati includono l'overhead architetturale di Spark, che su singola macchina rappresenta un costo non ammortizzato."

## Il problema di fondo

Spark è scritto sopra **Hadoop**, anche quando non usi HDFS o un cluster. La libreria `hadoop-common` è una dipendenza obbligatoria: contiene il codice per leggere/scrivere file, gestire permessi, fare path resolution, eccetera. Spark ci si appoggia per ogni operazione su filesystem (locale o distribuito che sia).

Hadoop è nato in ambiente Unix/Linux. Lì usa direttamente le **syscall POSIX**: `chmod`, `chown`, `stat`, link simbolici, permessi `rwxrwxrwx`, ownership utente/gruppo. Tutto questo esiste nativamente nel kernel Linux.

Windows non ha syscall POSIX. Ha le sue API (Win32), con un modello di permessi completamente diverso (ACL invece di rwx, SID invece di UID, eccetera). Quindi quando Hadoop su Windows deve fare un'operazione come "imposta i permessi 755 su questo file", il codice Java non sa come tradurla.

## La soluzione: `winutils.exe`

`winutils.exe` è un **piccolo eseguibile nativo Windows** che fa da ponte. Espone una serie di sottocomandi POSIX-like che Hadoop chiama come processi esterni:

```
winutils.exe chmod 755 C:\path\to\file
winutils.exe chown user:group C:\path\to\file
winutils.exe ls C:\path
winutils.exe systeminfo
```

Quando Hadoop (e quindi Spark) ha bisogno di un'operazione filesystem "in stile Unix", invece di provare a chiamare una syscall che non esiste, fa **`Runtime.exec("winutils.exe chmod ...")`** dietro le quinte. winutils traduce quella richiesta in chiamate Win32 native e restituisce il risultato in formato POSIX-like.

Insieme a `winutils.exe` viene di solito anche **`hadoop.dll`**, una libreria nativa che fa la stessa cosa ma chiamata via JNI (Java Native Interface) invece che via processo esterno. Più veloce ma usata in meno punti.

## Cosa succede se manca

Hadoop ha del codice di fallback per gestire l'assenza di winutils. Ma:

1. **Sparge warning a ogni operazione** (quello che vedi: `Did not find winutils.exe`). Non blocca, ma rallenta perché ogni log scrive su file.

2. **Le operazioni che servono permessi/ownership falliscono** o ritornano valori finti. Per esempio `setPermission()` su un file di shuffle: Hadoop si aspetta che il file abbia permessi 600 dopo la chiamata, ma senza winutils non ha modo di metterceli. Allora **ritenta**, oppure **emula in Java** con un fallback che è ordini di grandezza più lento, oppure **continua senza** ma marca quel file come "permessi indefiniti" e ogni successiva operazione su quel file paga controlli extra.

3. **Gli shuffle file vengono creati con path management più costoso**. Spark scrive **moltissimi** file durante uno shuffle (uno per partizione sorgente × partizione destinazione). Su un `groupByKey` con 128 partizioni puoi avere 128×128 = 16.384 file di shuffle. Su ognuno Hadoop tenta operazioni di permission/ownership. Su Linux: una syscall ciascuna, microsecondi totali. Su Windows senza winutils: fallback Java lento + eccezioni gestite + retry, **decine di millisecondi per file**. Per 16.384 file: **centinaia di secondi sprecati**.

4. **I file temporanei restano in giro più a lungo**. Senza winutils, Hadoop non riesce a fare cleanup affidabile delle directory temporanee → il disco si riempie di residui shuffle che richiedono di essere puliti a mano periodicamente.

## Perché ti pesa così tanto

Il tuo codice fa **tanti shuffle**:

- `groupByKey` sulle bande (uno shuffle per ogni `b` testato → 2 shuffle per ogni `n`)
- `distinct` sulle coppie candidate (un altro shuffle)
- implicito nelle action `collect` su RDD large

Per il preset sanity a n=1000 con 2 valori di b, sono **~6 shuffle**. Ognuno scrive migliaia di file. Senza winutils ogni file paga overhead di gestione → secondi che diventano minuti.

Per il preset full a n=1M, gli shuffle scrivono **centinaia di migliaia di file**, e l'overhead winutils-mancante diventa **devastante**: parliamo di ore di tempo perso. Non è il calcolo lento, è il file management Windows che è lento senza il ponte nativo.

## L'analogia

Pensala così: Hadoop è un viaggiatore inglese che parla solo inglese. winutils è il **traduttore** che assumi quando va in Italia. Senza traduttore, l'inglese riesce comunque a comunicare con gesti e segni: lentamente, con tanti malintesi, ripetendo le richieste, ma alla fine ottiene quello che vuole. Con il traduttore, è una conversazione fluida.

Spark/Hadoop su Windows senza winutils funzionano "a gesti": è quello che stai vedendo.

## Verifica veloce sul tuo sistema

Apri PowerShell e fai:

```powershell
where.exe winutils.exe
```

Se ti risponde con un percorso, l'hai. Se ti dice "INFO: Could not find files for the given pattern", non l'hai.

Anche:

```powershell
echo $env:HADOOP_HOME
```

Se è vuoto, la variabile non è settata.

Probabilmente entrambi i comandi non danno nulla → manca tutto. Questo è il singolo intervento con più impatto sui tempi che puoi fare ora.

## Una nota importante

`winutils.exe` non è distribuito ufficialmente da Apache Hadoop perché Apache non rilascia binari Windows. Viene compilato e pubblicato da contributor esterni della community (Steve Loughran, cdarlint, e altri). È **standard de facto**, sicuro nel senso che è open source e ispezionabile, ma formalmente è un binario di terze parti. Quasi tutti gli utenti Spark su Windows lo usano — non c'è alternativa.

apri <http://localhost:4040/> e guarda la sezione "Environment" → "System Properties". Se vedi `hadoop.home.dir` o `HADOOP_HOME` con un percorso, è lì che dovrebbe esserci winutils.exe. Se non c'è, è un segnale che non hai configurato correttamente l'ambiente Hadoop su Windows.
